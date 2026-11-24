from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
a=pd.read_csv(ROOT/'04_holdout_queue/holdout_lag.csv');b=pd.read_csv(ROOT/'04_holdout_queue/direct_hac_holdout.csv')
fig,ax=plt.subplots(1,2,figsize=(8,3.3),sharey=True)
for i,(d,ms,labels,title) in enumerate([(a,['fraction','hac'],['Fraction','Factorized'],'A: first test'),(b,['fraction','factorized','direct_masked'],['Fraction','Factorized','Direct masked'],'B: independent test')]):
 vals=[100*d.loc[d.model==m,'relative_error'].mean() for m in ms];ax[i].bar(labels,vals);ax[i].set_title(title);ax[i].set_ylim(0,24);ax[i].tick_params(axis='x',rotation=20);ax[i].bar_label(ax[i].containers[0],fmt='%.2f',padding=2,fontsize=9)
ax[0].set_ylabel('Mean absolute relative error (%)');fig.tight_layout();fig.savefig(OUT/'fig_variance.pdf',bbox_inches='tight');plt.close(fig)
d=pd.read_csv(ROOT/'06_udp_integration/socket_summary.csv');fig,ax=plt.subplots(1,2,figsize=(8,3.3));x=np.arange(3);w=.36
for j,(p,r) in enumerate([(0.1,0),(0.3,.8)]):
 z=d[(d.loss==p)&(d.persistence==r)].set_index('mode').loc[['none','xor','nack']];label='Independent 10%' if j==0 else 'Correlated 30%';ax[0].bar(x+(j-.5)*w,z.received*100,w,label=label);ax[1].bar(x+(j-.5)*w,z.masked_rmse*1e5,w,label=label)
for a in ax:a.set_xticks(x,['None','XOR','NACK']);a.legend(fontsize=8)
ax[0].set_ylabel('Reception (%)');ax[1].set_ylabel('TE RMSE ($10^{-5}$ bits)');fig.tight_layout();fig.savefig(OUT/'fig_udp.pdf',bbox_inches='tight');plt.close(fig)
