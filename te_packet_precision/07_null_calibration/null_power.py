"""Independent Monte Carlo null calibration and power for binary lag-one TE.
All inputs synthetic; no causal identification claim.
"""
from pathlib import Path
import csv, numpy as np
from scipy.stats import norm
N_CAL=800; N_TEST=800; ALPHA=.05; H=10
SCENARIOS=[('complete',0.,0.),('iid10',.1,0.),('iid30',.3,0.),('burst30',.3,.8)]
EFFECTS=[0.,.025,.05,.10]

def estimate(x,y,w):
    c=4*x[:,:-1]+2*y[:,:-1]+y[:,1:]; n=len(x)
    a=np.stack([((c==i)&w).sum(1) for i in range(8)],axis=1).reshape(n,2,2,2).astype(float)+.5
    p=a/a.sum((1,2,3),keepdims=True)
    return (p*np.log2(p*p.sum((1,3),keepdims=True)/(p.sum(3,keepdims=True)*p.sum(1,keepdims=True)))).sum((1,2,3))

def trials(seed,T,effect,n):
    rng=np.random.default_rng(seed); ans={ (s,m):[] for s,_,_ in SCENARIOS for m in ('mask','zero')}; fs={s:[] for s,_,_ in SCENARIOS}
    for start in range(0,n,100):
        b=min(100,n-start); x=(rng.random((b,T))<.2).astype(np.uint8); y=np.zeros_like(x); u=rng.random((b,T));y[:,0]=u[:,0]<.15
        for t in range(1,T): y[:,t]=u[:,t]<(.05+.6*y[:,t-1]+effect*x[:,t-1])
        for s,e,r in SCENARIOS:
            k=(T+H-1)//H; lost=np.zeros((b,k),bool); z=rng.random((b,k)); lost[:,0]=z[:,0]<e
            for j in range(1,k): lost[:,j]=z[:,j]<(e+r*(lost[:,j-1]-e))
            avail=np.repeat(~lost,H,axis=1)[:,:T]; w=avail[:,1:]&avail[:,:-1]
            ans[s,'mask'].extend(estimate(x,y,w)); ans[s,'zero'].extend(estimate(x*avail,y*avail,np.ones_like(w)));fs[s].extend(w.mean(1))
    return {k:np.array(v) for k,v in ans.items()}, {k:np.mean(v) for k,v in fs.items()}

def wilson(k,n):
    z=norm.ppf(.975); p=k/n; den=1+z*z/n; ctr=(p+z*z/(2*n))/den; half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return ctr-half,ctr+half

def main():
    rows=[]
    for i,T in enumerate((4000,12000)):
        cal,_=trials(75100+i,T,0,N_CAL);threshold={k:np.quantile(v,1-ALPHA,method='higher') for k,v in cal.items()}
        for j,effect in enumerate(EFFECTS):
            test,f=trials(75200+10*i+j,T,effect,N_TEST)
            for (s,m),v in test.items():
                for calibration in ('matched','complete'):
                    thr=threshold[s,m] if calibration=='matched' else threshold['complete','mask'];k=int((v>thr).sum());lo,hi=wilson(k,len(v));rows.append(dict(T=T,effect=effect,scenario=s,estimator=m,calibration=calibration,n_cal=N_CAL,n_test=N_TEST,alpha=ALPHA,threshold=thr,valid_fraction=f[s],reject=k,rate=k/len(v),wilson95_low=lo,wilson95_high=hi))
        print('finished',T,flush=True)
    out=Path(__file__).with_name('null_power.csv');
    with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
