"""Synthetic benchmark of transfer entropy variance under packet erasures."""
import numpy as np, csv, itertools, pathlib
OUT=pathlib.Path(__file__).parent
R=600; N=8000; H=10; K=400
states=np.array(list(itertools.product([0,1],repeat=3)))
def theory(a,b):
    Q=np.zeros((8,8))
    for i,(x,y,z) in enumerate(states):
        for j,(xx,yy,zz) in enumerate(states):
            if yy!=z: continue
            px=.3*(1-a)+a*x; py=.16+.18*z+b*xx
            Q[i,j]=(px if xx else 1-px)*(py if zz else 1-py)
    A=np.vstack((Q.T-np.eye(8),np.ones(8))); pi=np.linalg.lstsq(A,np.r_[np.zeros(8),1.],rcond=None)[0]
    assert np.all(pi>0) and np.allclose(pi@Q,pi)
    p=pi.reshape(2,2,2); score=np.log2(p*p.sum((0,2),keepdims=True)/(p.sum(2,keepdims=True)*p.sum(0,keepdims=True))).ravel()
    te=pi@score; psi=score-te; q=psi.copy(); gamma=[]
    for k in range(K+1):
        gamma.append((pi*psi)@q); q=Q@q
    return te,np.array(gamma),pi,score

def joint_w(p,rho,k):
    q=1-p; vals=[]
    for t in range(H):
        idx=sorted(set([(t-1)//H,t//H,(t+k-1)//H,(t+k)//H]))
        vals.append(q*np.prod([q+p*rho**(b-a) for a,b in zip(idx,idx[1:])]))
    return np.mean(vals)

def masks(rng,p,rho):
    if p==0: return np.ones((N,R),bool)
    u=rng.random((N//H,R)); good=np.empty(u.shape,bool); good[0]=u[0]>=p
    for i in range(1,len(good)): good[i]=u[i]<(1-p)*(1-rho)+rho*good[i-1]
    return np.repeat(good,H,axis=0)

def signals(rng,a,b):
    x=np.empty((N,R),bool);y=np.empty_like(x); x[0]=rng.random(R)<.3; y[0]=rng.random(R)<.3
    ux=rng.random((N,R));uy=rng.random((N,R))
    for t in range(1,N):
        x[t]=ux[t]<(.3*(1-a)+a*x[t-1]); y[t]=uy[t]<(.16+.18*y[t-1]+b*x[t-1])
    return x,y

def plugin(c,w):
    n=w.sum(0); counts=np.stack([((c==k)&w).sum(0) for k in range(8)]).reshape(2,2,2,-1)
    d=counts.sum((0,2),keepdims=True); a=counts.sum(2,keepdims=True); b=counts.sum(0,keepdims=True)
    with np.errstate(divide='ignore',invalid='ignore'):
        te=np.where(counts>0,counts*np.log2(counts*d/(a*b)),0).sum((0,1,2))/n
    return te

def main():
    rows=[]; raw=[]; rng=np.random.default_rng(20261125)
    for persistence,b in itertools.product([0.,.90],[.08,.25]):
        truth,gamma,pi,score=theory(persistence,b)
        L0=gamma[0]+2*gamma[1:].sum(); x,y=signals(rng,persistence,b); c=4*x[:-1].astype(np.uint8)+2*y[:-1].astype(np.uint8)+y[1:]
        for p,rho in [(0.,0.),(.1,0.),(.1,.8),(.3,0.),(.3,.8),(.3,.95)]:
            m=masks(rng,p,rho); w=m[:-1]&m[1:]; f=joint_w(p,rho,0); L=gamma[0]*f+2*sum(gamma[k]*joint_w(p,rho,k) for k in range(1,K+1)); t=N-1; v0=L0/(t*f); v1=L/(t*f*f)
            est=plugin(c,w); linear=truth+((score[c]-truth)*w).sum(0)/w.sum(0); vv=est.var(ddof=1); vl=linear.var(ddof=1)
            rows.append([persistence,b,p,rho,truth,f,w.mean(),est.mean(),vv,vl,v0,v1,abs(v0/vv-1),abs(v1/vv-1),abs(v0/vl-1),abs(v1/vl-1),gamma[0],L0,L])
            raw.extend([[persistence,b,p,rho,i,w[:,i].mean(),e,l] for i,(e,l) in enumerate(zip(est,linear))]);
        print('done',persistence,b,flush=True)
    headers=['x_persistence','coupling','loss','loss_persistence','true_te','f_theory','f_emp','mean_te','var_plugin','var_linear','var_f_only','var_longrun','err_f_plugin','err_L_plugin','err_f_linear','err_L_linear','gamma0','L0','L']
    for name,h,r in [('variance_summary.csv',headers,rows),('replicate_te.csv',['x_persistence','coupling','loss','loss_persistence','replicate','valid_fraction','te','linear_te'],raw)]:
        with open(OUT/name,'w',newline='') as o: wr=csv.writer(o);wr.writerow(h);wr.writerows(r)
if __name__=='__main__':main()
