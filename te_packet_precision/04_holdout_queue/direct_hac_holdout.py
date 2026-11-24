"""Compare direct masked HAC with factorized pilot HAC and fraction-only. Fresh seeds.
Training: complete pilot plus masks; validation: select each model bandwidth; test: untouched.
Run from any directory. Model errors are descriptive across reused scenarios.
"""
import csv
from pathlib import Path
import numpy as np
import validate as v
KS=(100,300,1000,2000); K=max(KS)
def pred(g,q,h,f,T,k):
    taper=1-np.arange(1,k+1)/(k+1)
    return [(g[0]+2*np.dot(taper,g[1:k+1]))/(T*f), (f*g[0]+2*np.dot(taper,g[1:k+1]*q[1:k+1]))/(T*f*f),(h[0]+2*np.dot(taper,h[1:k+1]))/(T*f*f)]
def main():
    rows=[]
    for j,rho in enumerate((.9,.98,.995)):
        ca=v.signal(84100+j,rho,64); va=v.signal(84200+j,rho,96); ev=v.signal(84300+j,rho,192)
        p=(np.bincount(ca.ravel(),minlength=8)+.5)/(ca.size+4); s=v.score(p); psi=s[ca]-p@s; T=len(psi);g=v.ac(psi,K)
        for i,(loss,r,D) in enumerate((p,r,d) for p in (.1,.3) for r in (0.,.8) for d in (50,100)):
            a=v.losses(85000+j*100+i,ca.shape[1],loss,r); b=v.losses(86000+j*100+i,va.shape[1],loss,r); c=v.losses(87000+j*100+i,ev.shape[1],loss,r)
            for mode in ('none','xor','nack'):
                wa,_=v.mask(a,mode,D); wb,_=v.mask(b,mode,D); wc,_=v.mask(c,mode,D); f=wa.mean();q=v.ac(wa.astype(float),K)/(1-np.arange(K+1)/T)
                # Standard amplitude-modulated score HAC: mean-center the observed score globally.
                z=wa*psi;z=z-z.mean();h=v.ac(z,K)
                pp={k:pred(g,q,h,f,T,k) for k in KS}; vv=v.te(va,wb).var(ddof=1); tv=v.te(ev,wc).var(ddof=1)
                for m,idx in [('fraction',0),('factorized',1),('direct_masked',2)]:
                    k=min(KS,key=lambda k:abs(np.log(max(pp[k][idx],1e-30)/vv))); est=pp[k][idx]
                    rows.append(dict(rho=rho,loss=loss,persistence=r,deadline_ms=D,mode=mode,model=m,selected_lag=k,predicted=est,validation_variance=vv,test_variance=tv,relative_error=abs(est/tv-1)))
        print('finished',rho,flush=True)
    with (Path(__file__).parent/'direct_hac_holdout.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
