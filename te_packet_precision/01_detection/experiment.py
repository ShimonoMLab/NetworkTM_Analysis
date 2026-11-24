import numpy as np, csv, pathlib
from itertools import product
OUT=pathlib.Path(__file__).parent
N=8000; P=80; S=12; lengths=[1000,2000,4000,8000]
def masks(seed,kind,mode,D):
 r=np.random.default_rng(seed); b=N//40; a=r.random(b*6); z=a<.1
 if kind=='burst':
  for k in range(1,len(z)):z[k]=a[k]<(.65 if z[k-1] else .1*.35/.9)
 z=z.reshape(b,6); ok=~z[:,:4]; rec=np.where(ok,np.arange(4)*10+20,999); m=(~ok).sum(1); j=(~ok).argmax(1); tx=np.full(b,4)
 if mode=='xor':
  tx+=1;k=np.where((m==1)&~z[:,4])[0];rec[k,j[k]]=60
 if mode=='nack':
  tx+=m>0;k=np.where((m>0)&~z[:,4])[0];tx[k]+=1;k=k[~z[k,5]];rec[k,j[k]]=90
 return np.repeat((rec<=np.arange(4)*10+D).ravel(),10),tx.mean()/4

def signals(seed):
 r=np.random.default_rng(seed);x=r.random((N,P))<.30;y=np.empty_like(x);y[0]=r.random(P)<.3;u=r.random((N,P));s=np.r_[np.zeros(P//2),np.full(P//2,.08)]
 for t in range(1,N):y[t]=u[t]<(.16+.18*y[t-1]+s*x[t-1])
 return x,y

def te(x,y,mask,fill):
 keep=mask[1:]&mask[:-1];f=keep.mean()
 if fill:x=x&mask[:,None];y=y&mask[:,None];keep=np.ones(len(mask)-1,bool)
 c=(4*y[1:].astype(np.uint8)+2*y[:-1]+x[:-1])[keep];m=len(c)
 q=np.stack([(c==k).sum(0) for k in range(8)]).reshape(2,2,2,-1).astype(float);a=q.sum(2,keepdims=True);b=q.sum(0,keepdims=True);d=q.sum((0,2),keepdims=True)
 with np.errstate(divide='ignore',invalid='ignore'):v=np.where(q>0,q*np.log2(q*d/(a*b)),0).sum((0,1,2))/m
 return v,f

def main():
 data={};meta={};raw=[]
 for split,off in [('cal',100),('test',1000)]:
  for i in range(S):
   x,y=signals(off+i)
   for kind,mode,D in list(product(['iid','burst'],['none','xor','nack'],[50,100]))+[('complete','none',0)]:
    mask,rate=masks(off+10000+i,kind,mode,D) if kind!='complete' else (np.ones(N,bool),1.)
    for n,fill in product(lengths,[False,True] if kind!='complete' else [False]):
     v,f=te(x[:n],y[:n],mask[:n],fill);key=(kind,mode,D,n,fill);data.setdefault((split,key),[]).append(v)
     if split=='test':meta.setdefault(key,[]).append([rate,f]);raw.append([i,*key,rate,f,*v])
  print(split,'done',flush=True)
 with open(OUT/'summary.csv','w') as f:
  w=csv.writer(f);w.writerow(['loss','mode','deadline_ms','n','zero_fill','rate','valid_fraction','threshold_bits','fpr','tpr','fpr_sd','tpr_sd','fpr_complete_threshold','tpr_complete_threshold'])
  for split,key in data:
   if split!='test':continue
   ar=np.array(data[('test',key)]);ca=np.array(data[('cal',key)]);thr=np.quantile(ca[:,:P//2],.99,method='higher');base=np.quantile(np.array(data[('cal',('complete','none',0,key[3],False))])[:,:P//2],.99,method='higher');fp=(ar[:,:P//2]>thr).mean(1);tp=(ar[:,P//2:]>thr).mean(1)
   w.writerow([*key,*np.mean(meta[key],axis=0),thr,fp.mean(),tp.mean(),fp.std(ddof=1),tp.std(ddof=1),(ar[:,:P//2]>base).mean(),(ar[:,P//2:]>base).mean()])
 with open(OUT/'per_seed_te.csv','w') as f:
  w=csv.writer(f);w.writerow(['seed_index','loss','mode','deadline_ms','n','zero_fill','rate','valid_fraction']+[f'te_pair{i}' for i in range(P)]);w.writerows(raw)
if __name__=='__main__':main()
