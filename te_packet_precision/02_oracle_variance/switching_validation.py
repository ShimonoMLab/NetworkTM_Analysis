"""Stationary Markov-modulated coupling benchmark with exact score covariance."""
import numpy as np, csv, itertools
import variance_validation as base
OUT=base.OUT; H=base.H; R=base.R; N=base.N; K=2000
states=np.array(list(itertools.product([0,1],repeat=4)))
def theory(rho):
    Q=np.zeros((16,16))
    for i,(z,x,y,v) in enumerate(states):
        for j,(zz,xx,yy,vv) in enumerate(states):
            if yy!=v:continue
            pz=.5*(1-rho)+rho*z; px=.3; py=.12+.18*v+(.02+.48*zz)*xx
            Q[i,j]=(pz if zz else 1-pz)*(px if xx else 1-px)*(py if vv else 1-py)
    A=np.vstack((Q.T-np.eye(16),np.ones(16)));pi=np.linalg.lstsq(A,np.r_[np.zeros(16),1.],rcond=None)[0]
    assert np.all(pi>0) and np.allclose(pi@Q,pi)
    p=pi.reshape(2,2,2,2).sum(0); score=np.log2(p*p.sum((0,2),keepdims=True)/(p.sum(2,keepdims=True)*p.sum(0,keepdims=True))).ravel(); te=p.ravel()@score; psi=np.tile(score-te,2); q=psi.copy(); gamma=[]
    for k in range(K+1):gamma.append((pi*psi)@q); q=Q@q
    return te,np.array(gamma),score

def signals(rng,rho):
    z=np.empty((N,R),bool);x=rng.random((N,R))<.3; y=np.empty_like(x);z[0]=rng.random(R)<.5;y[0]=rng.random(R)<.3;uz=rng.random((N,R));uy=rng.random((N,R))
    for t in range(1,N):
        z[t]=uz[t]<(.5*(1-rho)+rho*z[t-1]);y[t]=uy[t]<(.12+.18*y[t-1]+(.02+.48*z[t-1])*x[t-1])
    return x,y

def main():
    rng=np.random.default_rng(20261126); rows=[];raw=[]
    for rho in [.9,.98,.995]:
        truth,gamma,score=theory(rho); L0=gamma[0]+2*gamma[1:].sum();x,y=signals(rng,rho); c=4*x[:-1].astype(np.uint8)+2*y[:-1].astype(np.uint8)+y[1:]
        for p,r in [(0.,0.),(.1,0.),(.1,.8),(.3,0.),(.3,.8),(.3,.95)]:
            m=base.masks(rng,p,r);w=m[:-1]&m[1:];f=base.joint_w(p,r,0);L=gamma[0]*f+2*sum(gamma[k]*base.joint_w(p,r,k) for k in range(1,K+1));t=N-1;v0=L0/(t*f);v1=L/(t*f*f); est=base.plugin(c,w);lin=truth+((score[c]-truth)*w).sum(0)/w.sum(0);v=est.var(ddof=1);vl=lin.var(ddof=1)
            rows.append([rho,p,r,truth,f,w.mean(),est.mean(),v,vl,v0,v1,abs(v0/v-1),abs(v1/v-1),abs(v0/vl-1),abs(v1/vl-1),v1/v0,gamma[0],L0,L]); raw.extend([[rho,p,r,i,e,l] for i,(e,l) in enumerate(zip(est,lin))]);
        print('done',rho,flush=True)
    for name,h,data in [('switching_summary.csv',['state_persistence','loss','loss_persistence','true_te','f_theory','f_emp','mean_te','var_plugin','var_linear','var_f_only','var_longrun','err_f_plugin','err_L_plugin','err_f_linear','err_L_linear','ratio','gamma0','L0','L'],rows),('switching_replicates.csv',['state_persistence','loss','loss_persistence','replicate','te','linear_te'],raw)]:
        with open(OUT/name,'w',newline='') as f:wr=csv.writer(f);wr.writerow(h);wr.writerows(data)
if __name__=='__main__':main()
