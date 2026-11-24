"""Exploratory lag-bandwidth sensitivity on the same rho=.995 split, not new confirmation."""
import numpy as np
import validate as v
ca=v.signal(112,.995,96);ev=v.signal(902,.995,192);K=2000;lags=np.arange(K+1);p=(np.bincount(ca.ravel(),minlength=8)+.5)/(ca.size+4);s=v.score(p);g=v.ac(s[ca]-p@s,K);rows=[]
for i,(loss,rho,D) in enumerate((p,r,d) for p in (.1,.3) for r in (0.,.8) for d in (50,100)):
    zc=v.losses(2400+i,ca.shape[1],loss,rho);ze=v.losses(2500+i,ev.shape[1],loss,rho)
    for mode in ('none','xor','nack'):
        wc,_=v.mask(zc,mode,D);we,_=v.mask(ze,mode,D);f=wc.mean();q=v.ac(wc.astype(float),K)/(1-lags/len(wc));obs=v.te(ev,we).var(ddof=1)
        for k in (100,300,1000,2000):
            taper=1-lags[1:k+1]/(k+1);L0=g[0]+2*np.dot(g[1:k+1],taper);L=g[0]*f+2*np.dot(g[1:k+1]*q[1:k+1],taper);v0=L0/((v.N-1)*f);v1=L/((v.N-1)*f*f)
            rows.append(dict(loss=loss,persistence=rho,deadline_ms=D,mode=mode,lag=k,var_test=obs,var_fraction=v0,var_hac=v1,error_fraction=abs(v0/obs-1),error_hac=abs(v1/obs-1)))
v.write('lag_sensitivity.csv',rows)
