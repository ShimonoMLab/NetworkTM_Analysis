"""Pilot-estimated TE variance and policy selection. No physical network model."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.fft import rfft,irfft,next_fast_len
OUT=Path(__file__).parent
N=16000; K=300

def codes(x,y):return 4*x[:-1].astype(np.uint8)+2*y[:-1].astype(np.uint8)+y[1:]
def signal(seed,rho,r):
    g=np.random.default_rng(seed);n=N+500;z=g.random(r)<.5;x=g.random((n,r))<.3;y=np.empty_like(x);y[0]=g.random(r)<.3;u=g.random(x.shape);v=g.random(x.shape)
    for t in range(1,n):
        y[t]=u[t]<(.12+.18*y[t-1]+(.02+.48*z)*x[t-1]);z=v[t]<(.5*(1-rho)+rho*z)
    return codes(x[500:],y[500:])
def score(p):
    a=p.reshape((2,2,2)+p.shape[1:]);num=a*a.sum((0,2),keepdims=True);den=a.sum(2,keepdims=True)*a.sum(0,keepdims=True)
    with np.errstate(divide='ignore',invalid='ignore'):s=np.where((num>0)&(den>0),np.log2(num/den),0)
    return s.reshape(p.shape)
def te(c,w):
    q=np.stack([((c==i)&w).sum(0) for i in range(8)]).astype(float);q/=q.sum(0);return (q*score(q)).sum(0)
def ac(a,k=None):
    if k is None:k=K
    n=len(a);m=next_fast_len(2*n-1);a=rfft(a,n=m,axis=0);return irfft(a*a.conj(),n=m,axis=0)[:k+1].mean(1)/n

def losses(seed,r,p,rho):
    g=np.random.default_rng(seed);u=g.random((N//40*6,r));z=np.empty(u.shape,bool);z[0]=u[0]<p
    for t in range(1,len(z)):z[t]=u[t]<p*(1-rho)+rho*z[t-1]
    return z.reshape(N//40,6,r)
def mask(z,mode,D):
    # 4 data slots, 2 reserved repair slots, 10ms spacing, forward delay 10ms,
    # reverse delay 5ms. Feedback sent at slot 4 (40ms), reaches sender at 45ms.
    # One equal-sized control packet and at most one retry; all counted.
    b,_,r=z.shape;ok=~z[:,:4];lost=(~ok).sum(1);j=(~ok).argmax(1)
    ar=np.where(ok,np.arange(4)[None,:,None]*10+10,np.inf);sent=np.full((b,r),4.)
    if mode=='xor':
        sent+=1;i,h=np.where((lost==1)&~z[:,4]);ar[i,j[i,h],h]=50
    if mode=='nack':
        sent+=lost>0;retry=(lost>0)&~z[:,4];sent+=retry;i,h=np.where(retry&~z[:,5]);ar[i,j[i,h],h]=60
    m=np.repeat((ar<=np.arange(4)[None,:,None]*10+D).reshape(b*4,r),10,axis=0);return m[:-1]&m[1:],sent.mean()/4

def run(ca,ev,label,seed):
    p=(np.bincount(ca.ravel(),minlength=8)+.5)/(ca.size+4);s=score(p);mu=p@s;gamma=ac(s[ca]-mu);lags=np.arange(K+1);tap=1-lags/(K+1);L0=gamma[0]+2*np.dot(gamma[1:],tap[1:]);base=te(ev,np.ones_like(ev,bool));rows=[];pol=[]
    for i,(loss,rho,D) in enumerate((p,r,d) for p in (.1,.3) for r in (0.,.8) for d in (50,100)):
        zc=losses(seed+i,ca.shape[1],loss,rho);ze=losses(seed+100+i,ev.shape[1],loss,rho);group=[]
        for mode in ('none','xor','nack'):
            wc,rate=mask(zc,mode,D);we,rate_e=mask(ze,mode,D);f=wc.mean();q=ac(wc.astype(float))/(1-lags/len(wc));L=gamma[0]*q[0]+2*np.dot(gamma[1:]*q[1:],tap[1:]);v0=L0/((N-1)*f);v1=L/((N-1)*f*f);e=te(ev,we);v=e.var(ddof=1)
            row=dict(dataset=label,loss=loss,persistence=rho,deadline_ms=D,mode=mode,pilot_TE=mu,pilot_f=f,test_f=we.mean(),pilot_rate=rate,test_rate=rate_e,var_fraction=v0,var_hac=v1,var_test=v,error_fraction=abs(v0/v-1),error_hac=abs(v1/v-1),mean_te=e.mean(),rmse_vs_complete=np.sqrt(np.mean((e-base)**2)),pilot_n=ca.shape[1],test_n=ev.shape[1]);rows.append(row);group.append(row)
        for model in ('var_fraction','var_hac'):
            g=min(group,key=lambda x:x[model]);best=min(group,key=lambda x:x['var_test']);pol.append(dict(dataset=label,loss=loss,persistence=rho,deadline_ms=D,model=model,selected=g['mode'],test_best=best['mode'],test_variance_ratio=g['var_test']/best['var_test'],test_variance=g['var_test']))
    return rows,pol

def asdf(path,bin_ms=1,index_base=1):
    m=loadmat(path);names=[k for k,v in m.items() if not k.startswith('__') and isinstance(v,np.ndarray) and v.dtype==object];assert len(names)==1,names
    cells=m[names[0]].ravel();T=int(np.asarray(cells[-1]).ravel()[1]);spikes=[np.asarray(c).ravel().astype(np.int64)-index_base for c in cells[:-2]];length=T//bin_ms;B=length//N;split=B//2;assert min(split,B-split)>=8,'Need eight blocks per split';counts=np.array([np.sum((s>=0)&(s<split*N*bin_ms)) for s in spikes]);a,b=np.argsort(-counts)[:2];X=np.zeros(B*N,bool);Y=np.zeros_like(X)
    for arr,i in ((X,a),(Y,b)):
        ix=spikes[i]//bin_ms;arr[ix[(ix>=0)&(ix<len(arr))]]=1
    c=codes(X.reshape(B,N).T,Y.reshape(B,N).T);meta=dict(file=Path(path).name,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),neurons=len(spikes),duration_ms=T,pair=[int(a),int(b)],bin_ms=bin_ms,index_base=index_base,blocks=B,pilot_blocks=split,warning='Within-record blocks may be dependent/nonstationary; descriptive single-pair pilot only. Check ASDF indexing and bin units. No anatomical ground truth. Packet length is ten binned samples; timing is illustrative for bin_ms !=1.')
    return c[:,:split],c[:,split:],meta

def write(name,rows):
    with open(OUT/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    global K
    ap=argparse.ArgumentParser();ap.add_argument('--asdf');ap.add_argument('--bin-ms',type=int,default=1);ap.add_argument('--index-base',type=int,choices=[0,1],default=1);ap.add_argument('--lag',type=int,default=300);a=ap.parse_args();K=a.lag;assert 1<=K<N-1;rows=[];pol=[]
    if a.asdf:
        ca,ev,meta=asdf(a.asdf,a.bin_ms,a.index_base);(OUT/'real_metadata.json').write_text(json.dumps(meta,indent=2));rows,pol=run(ca,ev,'recorded_ASDF',200);prefix='real_'
    else:
        for i,rho in enumerate((.9,.98,.995)):
            ca=signal(110+i,rho,96);ev=signal(900+i,rho,192);r,p=run(ca,ev,f'synthetic_rho_{rho}',400+i*1000);rows+=r;pol+=p;print('done',rho,flush=True)
        prefix='synthetic_'
    write(prefix+'variance.csv',rows);write(prefix+'policy.csv',pol)
if __name__=='__main__':main()
