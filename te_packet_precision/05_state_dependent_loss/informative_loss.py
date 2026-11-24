"""Independent pilot/test validation: mask-aware TE under observable-state dependent loss.
Synthetic data only. Reliable per-packet state metadata is assumed, not implemented.
Conditional independent packet reception is assumed for propensity weighting.
"""
import numpy as np, csv
from pathlib import Path
N=40000; H=10; B=N//H; R=128

def generate(seed, reps, scenario):
    g=np.random.default_rng(seed)
    z=np.empty((B,reps),bool);z[0]=g.random(reps)<.5
    u=g.random((B,reps))
    for i in range(1,B):z[i]=u[i]<(.025+.95*z[i-1])
    state=np.repeat(z,H,axis=0)
    x=g.random((N,reps))<np.where(state,.4,.08)
    y=np.empty_like(x);y[0]=False;u=g.random(x.shape)
    for t in range(1,N):y[t]=u[t]<(.06+.15*y[t-1]+np.where(state[t],.5,.10)*x[t-1])
    code=4*x[:-1].astype(np.uint8)+2*y[:-1].astype(np.uint8)+y[1:]
    if scenario=='mcar': p=np.full(z.shape,.7)
    elif scenario=='observed_state': p=np.where(z,.45,.95)
    else: p=np.where(x.reshape(B,H,reps).sum(1)>=3,.45,.95)
    received=g.random(p.shape)<p
    return z,received,code

def te(c,w):
    counts=np.stack([(w*(c==i)).sum(0) for i in range(8)]);a=(counts+.5)/(counts.sum(0)+4);a=a.reshape(2,2,2,-1)
    v=a*a.sum((0,2),keepdims=True)/(a.sum(2,keepdims=True)*a.sum(0,keepdims=True));return (a*np.log2(v)).sum((0,1,2))

def propensity(z,m):return np.array([m[z==k].mean() for k in (False,True)])
def calc(z,m,c,p):
    r=np.repeat(m,H,axis=0);w=r[:-1]&r[1:];pp=np.repeat(p[z.astype(int)],H,axis=0);pair=pp[:-1].copy();pair[H-1::H]*=pp[H::H];
    return [te(c,np.ones_like(w)),te(c,w),te(c,w/pair)]

def main():
    rows=[];allrows=[]
    for i,s in enumerate(['mcar','observed_state','unobserved_activity']):
        z,m,c=generate(84000+i,64,s);p=propensity(z,m)
        z,m,c=generate(85000+i,R,s);full,cc,ipw=calc(z,m,c,p)
        for name,values in [('complete_case',cc),('state_ipw',ipw)]:
            err=values-full
            rows.append(dict(scenario=s,method=name,n=R,received=m.mean(),propensity_0=p[0],propensity_1=p[1],full_te=full.mean(),bias=err.mean(),rmse=np.sqrt(np.mean(err**2))))
            allrows.extend(dict(scenario=s,method=name,replicate=j,full_te=full[j],estimate=values[j],error=err[j]) for j in range(R))
        g=np.random.default_rng(86000+i);ix=g.integers(R,size=(4000,R));delta=np.sqrt(np.mean((ipw-full)[ix]**2,axis=1))-np.sqrt(np.mean((cc-full)[ix]**2,axis=1));print(s,rows[-2:], 'paired RMSE difference 95% interval', np.quantile(delta,[.025,.975]),flush=True)
        rows[-1].update(rmse_diff_ci_low=np.quantile(delta,.025),rmse_diff_ci_high=np.quantile(delta,.975));rows[-2].update(rmse_diff_ci_low='',rmse_diff_ci_high='')
    for name,data in [('informative_summary.csv',rows),('informative_replicates.csv',allrows)]:
        with (Path(__file__).parent/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
if __name__=='__main__':main()
