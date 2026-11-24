"""Two Linux network namespaces over veth; independent entropy TE check.
Same physical host; no physical LAN or real-network-loss claim.
"""
import argparse,csv,hashlib,json,os,sys,tempfile,time,subprocess,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'06_udp_integration'))
import udp_te as u
from synthetic_benchmark import synthetic

def entropy(*arrays):
    z=np.zeros(len(arrays[0]),np.int64)
    for a in arrays: z=2*z+a
    counts=np.bincount(z,minlength=2**len(arrays)); p=counts[counts>0]/len(z)
    return -sum(float(t)*math.log2(float(t)) for t in p)

def ref_te(a,m,ch):
    w=m[:-1]&m[1:]; assert w.any()
    out=[]
    for i in ch:
        for j in ch:
            if i==j: continue
            x=a[:-1,i][w]; y=a[:-1,j][w]; z=a[1:,j][w]
            out.append(entropy(x,y)+entropy(y,z)-entropy(y)-entropy(x,y,z))
    return np.asarray(out)

def main():
    p=argparse.ArgumentParser();p.add_argument('--peer',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    arc=list(csv.DictReader((ROOT/'06_udp_integration/socket_results.csv').open()));rows=[]
    with tempfile.TemporaryDirectory() as d:
        for name,seed,rate in [('synthetic_low.npy',74001,.0025),('synthetic_high.npy',74002,.0055)]:
            x=synthetic(seed,rate);source=Path(d)/name;np.save(source,x)
            meta=next(v for v in json.loads((ROOT/'06_udp_integration/metadata.json').read_text()) if v['file']==name)
            assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['sha256']
            ch=meta['channels']; y=x[20000:];g=len(y)//40;ref=ref_te(y,np.ones(len(y),bool),ch)
            for r in [r for r in arc if r['file']==name]:
                p0,rho,rep,mode=float(r['loss']),float(r['persistence']),int(r['rep']),r['mode'];out=Path(d)/'rx.npz';out.unlink(missing_ok=True);port=50100
                cmd=['nsenter','-t',a.peer,'-n',sys.executable,str(ROOT/'06_udp_integration/udp_te.py'),'receive','--bind','10.240.0.2','--port',str(port),'--groups',str(g),'--mode',mode,'--loss',str(p0),'--persistence',str(rho),'--seed',str(73100+rep),'--out',str(out)]
                proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                try:
                    for _ in range(150):
                        if proc.poll() is not None: raise RuntimeError(proc.communicate())
                        check=subprocess.check_output(['nsenter','-t',a.peer,'-n','ss','-uln'],text=True)
                        if f'10.240.0.2:{port}' in check: break
                        time.sleep(.02)
                    else: raise RuntimeError('receiver did not bind')
                    c=u.send(y,'10.240.0.2',port,mode);so,se=proc.communicate(timeout=15);assert proc.returncode==0,(so,se)
                finally:
                    if proc.poll() is None: proc.kill();proc.wait()
                b=np.load(out,allow_pickle=False);m=b['mask'];data=b['data'];assert np.array_equal(data[m],y[m]);assert c['tx']==int(b['rx']) and c['rx']==int(b['tx']) and c['bt']==int(b['br']) and c['br']==int(b['bt'])
                full=ref_te(data,np.ones(len(data),bool),ch);masked=ref_te(data,m,ch);native=u.te(data,m,ch)
                rr=float(np.sqrt(np.mean((masked-ref)**2)));zr=float(np.sqrt(np.mean((full-ref)**2)));diff=float(np.max(np.abs(masked-native)))
                for k,v in {'received':m.mean(),'valid':(m[:-1]&m[1:]).mean(),'masked_rmse':rr,'zero_rmse':zr,**{k:c[k] for k in ('tx','rx','bt','br','repair')}}.items():assert abs(float(r[k])-v)<1e-12,(k,r[k],v)
                assert diff<1e-12
                rows.append(dict(file=name,loss=p0,persistence=rho,rep=rep,mode=mode,received=m.mean(),valid=(m[:-1]&m[1:]).mean(),masked_rmse=rr,zero_rmse=zr,te_max_abs_difference_bits=diff,payload_match=True,archive_match=True,**c))
                print(f'{len(rows)}/48 {name} {p0} {rep} {mode}',flush=True)
    assert len(rows)==48
    with open(a.output,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={'scope':'Two Linux network namespaces connected by virtual Ethernet on one physical host; application-injected loss','python':sys.version.split()[0],'numpy':np.__version__,'sender_network_namespace':os.readlink('/proc/self/ns/net'),'receiver_network_namespace':os.readlink(f'/proc/{a.peer}/ns/net'),'trials':len(rows),'all_archive_match':all(r['archive_match'] for r in rows),'max_te_difference_bits':max(r['te_max_abs_difference_bits'] for r in rows)}
    Path(a.output).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
