from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
here=Path(__file__).resolve().parent
d=pd.read_csv(here/'null_power.csv')
fig,ax=plt.subplots(figsize=(6.2,3.5))
s=d[(d['T']==12000)&(d.effect==0)&(d.scenario!='complete')]
sc=['iid10','iid30','burst30']; x=np.arange(3); w=.24
for j,(est,cal,label) in enumerate([('mask','complete','Mask, complete threshold'),('zero','complete','Zero fill, complete threshold'),('mask','matched','Mask, matched threshold')]):
    q=s[(s.estimator==est)&(s.calibration==cal)].set_index('scenario').loc[sc]
    ax.bar(x+(j-1)*w,q.rate*100,w,label=label,yerr=np.array([q.rate-q.wilson95_low,q.wilson95_high-q.rate])*100,capsize=2)
ax.axhline(5,color='black',ls='--',lw=1,label='Nominal 5%')
ax.set_xticks(x,['Independent 10%','Independent 30%','Correlated 30%']);ax.set_ylabel('False-positive rate (%)');ax.set_ylim(0,100);ax.legend(fontsize=7,loc='upper left');fig.tight_layout();fig.savefig(here/'fig_null_calibration.pdf');plt.close(fig)
