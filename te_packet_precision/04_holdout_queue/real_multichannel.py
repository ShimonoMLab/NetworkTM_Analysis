"""Descriptive within-record multi-pair pilot; no anatomical truth assumed.
Requires original ASDF .mat supplied by data owner. Never copies raw data to outputs.
"""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import validate as v

def load_pairs(path,base,block_cap):
    m=loadmat(path); names=[k for k,x in m.items() if not k.startswith('__') and isinstance(x,np.ndarray) and x.dtype==object]; assert len(names)==1,names
    c=m[names[0]].ravel(); T=int(np.asarray(c[-1]).ravel()[1]); sp=[np.asarray(x).ravel().astype(np.int64)-base for x in c[:-2]]
    B=T//v.N; cut=B//2; assert cut>=8 and B-cut>=8,'Need at least eight blocks per half'; assert len(sp)>=4
    train=np.linspace(0,cut-1,min(cut,block_cap),dtype=int); test=np.linspace(cut,B-1,min(B-cut,block_cap),dtype=int)
    # Select channels from pilot half only, based on counts in first half.
    counts=[np.count_nonzero((s>=0)&(s<cut*v.N)) for s in sp]; top=np.argsort(-np.asarray(counts),kind='stable')[:4]; ids=np.r_[train,test]; arr={}
    for i in top:
        x=np.zeros((v.N,len(ids)),bool); s=sp[i]; s=s[(s>=0)&(s<T)]; blocks=s//v.N; where=np.searchsorted(ids,blocks); ok=where<len(ids); ok[ok]=ids[where[ok]]==blocks[ok]; x[s[ok]%v.N,where[ok]]=True; arr[int(i)]=x
    pairs=[(int(a),int(b)) for a in top for b in top if a!=b]
    meta=dict(file=Path(path).name,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),duration_bins=T,neurons=len(sp),index_base=base,block_length=v.N,pilot_block_indices=train.tolist(),test_block_indices=test.tolist(),top_channels=top.tolist(),pairs=pairs,warning='One recording; blocks may be dependent/nonstationary; no structural ground truth; high-rate channel selection only; no generalization across recordings.')
    return arr,pairs,len(train),meta

def main():
    p=argparse.ArgumentParser(); p.add_argument('asdf');p.add_argument('--index-base',type=int,choices=[0,1],required=True);p.add_argument('--block-cap',type=int,default=32);p.add_argument('--lag',type=int,default=1000);p.add_argument('--out',default='real_results'); a=p.parse_args();assert a.block_cap>=8 and 1<=a.lag<v.N-1;v.K=a.lag
    arr,pairs,n,meta=load_pairs(a.asdf,a.index_base,a.block_cap);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'metadata.json').write_text(json.dumps(meta,indent=2)); rr=[];pp=[]
    for i,(x,y) in enumerate(pairs):
        c=v.codes(arr[x],arr[y]); r,p=v.run(c[:,:n],c[:,n:],f'{x}->{y}',78000+i*1000);rr+=r;pp+=p
    for name,rows in [('variance.csv',rr),('policy.csv',pp)]:
        with (out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
