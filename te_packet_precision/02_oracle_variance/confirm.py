"""Independent confirmation with longer records and a fresh random seed."""
import csv, numpy as np
import variance_validation as b
import switching_validation as s
b.N=s.N=16000; b.R=s.R=1000
rng=np.random.default_rng(726143); rows=[]; rho=.98; truth,gamma,score=s.theory(rho); L0=gamma[0]+2*gamma[1:].sum(); x,y=s.signals(rng,rho); c=4*x[:-1].astype(np.uint8)+2*y[:-1].astype(np.uint8)+y[1:]
for p,r in [(0.,0.),(.3,0.),(.3,.8),(.3,.95)]:
    m=b.masks(rng,p,r); w=m[:-1]&m[1:]; f=b.joint_w(p,r,0);L=gamma[0]*f+2*sum(gamma[k]*b.joint_w(p,r,k) for k in range(1,len(gamma))); v0=L0/(b.N-1)/f; v1=L/(b.N-1)/f**2; e=b.plugin(c,w);v=e.var(ddof=1); bootrng=np.random.default_rng(4321); boot=e[bootrng.integers(0,len(e),(4000,len(e)))].var(axis=1,ddof=1);lo,hi=np.quantile(boot,[.025,.975]);rows.append([rho,p,r,b.N,b.R,w.mean(),v,lo,hi,v0,v1,abs(v0/v-1),abs(v1/v-1)])
with open(b.OUT/'confirmation.csv','w',newline='') as f:
    wr=csv.writer(f);wr.writerow(['state_persistence','loss','loss_persistence','n','replicates','f_emp','var_plugin','var_ci_low','var_ci_high','var_f_only','var_longrun','err_f_plugin','err_L_plugin']);wr.writerows(rows)
for row in rows:print(row)
