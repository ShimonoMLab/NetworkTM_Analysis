"""Select lag on validation data; test only on fresh independent trajectories."""
import numpy as np, csv
from pathlib import Path
import validate as v
KS=(100,300,1000,2000); KMAX=max(KS)
def fit(c):
    p=(np.bincount(c.ravel(),minlength=8)+.5)/(c.size+4); s=v.score(p)
    return v.ac(s[c]-p@s,KMAX)
def calc(g,w):
    lag=np.arange(KMAX+1); f=w.mean(); q=v.ac(w.astype(float),KMAX)/(1-lag/len(w)); out={}
    for k in KS:
        t=1-lag[1:k+1]/(k+1)
        out[k]=((g[0]+2*np.dot(g[1:k+1],t))/((v.N-1)*f),(g[0]*f+2*np.dot(g[1:k+1]*q[1:k+1],t))/((v.N-1)*f*f))
    return out
def main():
    rows=[]
    for j,rho in enumerate((.9,.98,.995)):
        ca=v.signal(71100+j,rho,48); va=v.signal(71200+j,rho,64); ev=v.signal(71300+j,rho,96); g=fit(ca)
        for i,(p,r,D) in enumerate((p,r,d) for p in (.1,.3) for r in (0.,.8) for d in (50,100)):
            a=v.losses(72000+j*100+i,ca.shape[1],p,r); b=v.losses(73000+j*100+i,va.shape[1],p,r); c=v.losses(74000+j*100+i,ev.shape[1],p,r)
            for mode in ('none','xor','nack'):
                wa,_=v.mask(a,mode,D); wb,_=v.mask(b,mode,D); wc,_=v.mask(c,mode,D)
                pred=calc(g,wa); val=v.te(va,wb).var(ddof=1); test=v.te(ev,wc).var(ddof=1)
                for model,kidx in [('fraction',0),('hac',1)]:
                    k=min(KS,key=lambda k:abs(np.log(max(pred[k][kidx],1e-30)/val)))
                    est=pred[k][kidx];rows.append(dict(rho=rho,loss=p,persistence=r,deadline_ms=D,mode=mode,model=model,selected_lag=k,predicted=est,validation_variance=val,test_variance=test,relative_error=abs(est/test-1)))
        print('finished',rho,flush=True)
    with open(Path(__file__).parent/'holdout_lag.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
