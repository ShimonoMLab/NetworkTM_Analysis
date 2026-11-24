import pandas as pd, numpy as np
from pathlib import Path
p=Path(__file__).parent;s=pd.read_csv(p/'summary.csv');r=pd.read_csv(p/'per_seed_te.csv'); rng=np.random.default_rng(20261124); res=[]
for loss in ['iid','burst']:
 for D in [50,100]:
  for mode in ['xor','nack']:
   a=[]
   for m in ['none',mode]:
    q=r[(r.loss==loss)&(r['mode']==m)&(r.deadline_ms==D)&(r.n==2000)&(~r.zero_fill)].sort_values('seed_index');th=s[(s.loss==loss)&(s['mode']==m)&(s.deadline_ms==D)&(s.n==2000)&(~s.zero_fill)].threshold_bits.iloc[0];a.append((q[[f'te_pair{i}' for i in range(40,80)]].values>th).mean(1))
   d=a[1]-a[0];boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1);res.append([loss,D,mode,d.mean(),*np.quantile(boot,[.025,.975])])
pd.DataFrame(res,columns=['loss','deadline_ms','mode','tpr_diff','ci_low','ci_high']).to_csv(p/'paired_bootstrap.csv',index=False)
print(pd.DataFrame(res).round(4));print('T80 interpolation:')
base=s[(s.loss=='complete')&(~s.zero_fill)].set_index('n'); t0=1000+1000*(.8-base.loc[1000].tpr)/(base.loc[2000].tpr-base.loc[1000].tpr)
for m in ['none','xor','nack']:
 a=s[(s.loss=='iid')&(s['mode']==m)&(s.deadline_ms==100)&(~s.zero_fill)].set_index('n'); t=1000+1000*(.8-a.loc[1000].tpr)/(a.loc[2000].tpr-a.loc[1000].tpr);print(m,t,t0/a.loc[2000].valid_fraction)
