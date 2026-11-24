"""Controlled UDP payload reconstruction and TE benchmark. Local endpoints, artificial erasures, no real-time deadlines. Raw input data are never redistributed."""
import argparse,csv,hashlib,json,socket,struct,threading,time
from pathlib import Path
import numpy as np
H=struct.Struct('!BIH'); N=10; C=128; SIZE=160

def te(a,m,channels):
    w=m[:-1]&m[1:]; out=[]
    for i in channels:
        for j in channels:
            if i==j:continue
            c=4*a[:-1,i]+2*a[:-1,j]+a[1:,j]; q=np.bincount(c[w],minlength=8).reshape(2,2,2).astype(float); q/=q.sum(); n=q*q.sum((0,2),keepdims=True); d=q.sum(2,keepdims=True)*q.sum(0,keepdims=True); z=q>0;out.append(np.sum(q[z]*np.log2(n[z]/d[z])))
    return np.array(out)
def losses(seed,g,p,rho):
    rng=np.random.default_rng(seed);u=rng.random(g*6);z=np.empty(len(u),bool);z[0]=u[0]<p
    for i in range(1,len(u)):z[i]=u[i]<p*(1-rho)+rho*z[i-1]
    return z.reshape(g,6)
def receiver(sock,g,mode,z,out):
    payload=np.zeros((g,4,SIZE),np.uint8); m=np.zeros((g,4),bool); parity=np.zeros((g,SIZE),np.uint8); parok=np.zeros(g,bool);rx=tx=br=bt=0
    try:
        while True:
            b,addr=sock.recvfrom(65535); k,i,j=H.unpack(b[:H.size]);rx+=1;br+=len(b)
            if k==4:
                reply=H.pack(5,i,65535);sock.sendto(reply,addr);tx+=1;bt+=len(reply);break
            if k in (0,3):
                if not z[i,j if k==0 else 5]:payload[i,j]=np.frombuffer(b[H.size:],np.uint8);m[i,j]=True
            elif k==1:
                if not z[i,4]:parity[i]=np.frombuffer(b[H.size:],np.uint8);parok[i]=True
            elif k==2:
                missing=np.flatnonzero(~m[i]);idx=65535
                if mode=='xor' and len(missing)==1 and parok[i]:
                    j=missing[0];payload[i,j]=parity[i]^np.bitwise_xor.reduce(payload[i,m[i]],axis=0);m[i,j]=True
                if mode=='nack' and len(missing) and not z[i,4]:idx=int(missing[0])
                reply=H.pack(5,i,idx);sock.sendto(reply,addr);tx+=1;bt+=len(reply)
        out.update(data=np.unpackbits(payload.reshape(-1,SIZE),axis=1).reshape(-1,C),mask=np.repeat(m.ravel(),N),rx=rx,tx=tx,br=br,bt=bt)
    except Exception as e:out['error']=repr(e)
    finally:sock.close()
def send(a,host,port,mode):
    g=len(a)//40;pack=np.packbits(a[:g*40].reshape(g,4,N*C),axis=2); sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);sock.settimeout(10);tx=rx=bt=br=repair=0;start=time.monotonic()
    def emit(k,i,j,d=b''):
        nonlocal tx,bt
        b=H.pack(k,i,j)+d;sock.sendto(b,(host,port));tx+=1;bt+=len(b)
    def ack(i):
        nonlocal rx,br
        b,_=sock.recvfrom(1024);rx+=1;br+=len(b);k,g,j=H.unpack(b);assert k==5 and g==i;return j
    try:
        for i in range(g):
            for j in range(4):emit(0,i,j,pack[i,j].tobytes())
            if mode=='xor':emit(1,i,0,np.bitwise_xor.reduce(pack[i],axis=0).tobytes());repair+=1
            emit(2,i,0);j=ack(i)
            if mode=='nack' and j!=65535:emit(3,i,j,pack[i,j].tobytes());repair+=1
        emit(4,g,0);ack(g)
    finally:sock.close()
    return dict(tx=tx,rx=rx,bt=bt,br=br,repair=repair,elapsed_s=time.monotonic()-start)
def read(path):
    a=np.load(path,allow_pickle=False);assert a.ndim==2 and a.shape[1]==C and np.all((a==0)|(a==1));return a.astype(np.uint8)
def benchmark(paths,out,reps):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];meta=[]
    for path in paths:
        x=read(path);cut=len(x)//80*40;channels=np.argsort(-x[:cut].sum(0),kind='stable')[:8];a=x[cut:2*cut];g=len(a)//40;full=te(a,np.ones(len(a),bool),channels);meta.append(dict(file=Path(path).name,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),shape=list(x.shape),pilot_samples=cut,test_samples=len(a),channels=channels.tolist(),provenance='User binary spike/event array; recording origin and time step unverified'))
        for p,rho in ((.1,0),(.3,.8)):
            for rep in range(reps):
                for mode in ('none','xor','nack'):
                    z=losses(73100+rep,g,p,rho);sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);sock.bind(('127.0.0.1',0));sock.settimeout(10);r={};th=threading.Thread(target=receiver,args=(sock,g,mode,z,r));th.start();c=send(a,'127.0.0.1',sock.getsockname()[1],mode);th.join(12);assert not th.is_alive() and 'error' not in r,r.get('error');b,m=r['data'],r['mask'];assert c['tx']==r['rx'] and c['bt']==r['br'] and c['br']==r['bt'];assert np.array_equal(b[m],a[m]);e=te(b,m,channels);zero=te(b,np.ones(len(b),bool),channels)
                    rows.append(dict(file=Path(path).name,loss=p,persistence=rho,rep=rep,mode=mode,pairs=len(full),received=m.mean(),valid=(m[:-1]&m[1:]).mean(),masked_rmse=np.sqrt(np.mean((e-full)**2)),zero_rmse=np.sqrt(np.mean((zero-full)**2)),payload_match=True,**c))
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));
    with (out/'socket_results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True);b=s.add_parser('benchmark');b.add_argument('inputs',nargs='+');b.add_argument('--out',default='results');b.add_argument('--reps',type=int,default=4);r=s.add_parser('receive');r.add_argument('--port',type=int,default=50050);r.add_argument('--bind',default='127.0.0.1');r.add_argument('--groups',type=int,required=True);r.add_argument('--mode',choices=['none','xor','nack'],required=True);r.add_argument('--seed',type=int,default=73100);r.add_argument('--loss',type=float,default=.1);r.add_argument('--persistence',type=float,default=0);r.add_argument('--out',default='received.npz');t=s.add_parser('send');t.add_argument('input');t.add_argument('--host',required=True);t.add_argument('--port',type=int,default=50050);t.add_argument('--mode',choices=['none','xor','nack'],required=True);e=s.add_parser('evaluate');e.add_argument('input');e.add_argument('received');a=p.parse_args()
    if a.cmd=='benchmark':benchmark(a.inputs,a.out,a.reps)
    elif a.cmd=='receive':
        sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);sock.bind((a.bind,a.port));sock.settimeout(60);r={};receiver(sock,a.groups,a.mode,losses(a.seed,a.groups,a.loss,a.persistence),r);assert 'error' not in r,r.get('error');np.savez_compressed(a.out,**r)
    elif a.cmd=='send':print(json.dumps(send(read(a.input),a.host,a.port,a.mode),indent=2))
    else:
        x=read(a.input);r=np.load(a.received,allow_pickle=False);m=r['mask'];b=r['data'];x=x[:len(b)];ch=np.argsort(-x[:len(x)//2].sum(0),kind='stable')[:8];cut=len(x)//2;assert np.array_equal(b[m],x[m]);ref=te(x[cut:],np.ones(len(x)-cut,bool),ch);val=te(b[cut:],m[cut:],ch);print(dict(received=float(m[cut:].mean()),rmse=float(np.sqrt(np.mean((ref-val)**2))),payload_match=True))
