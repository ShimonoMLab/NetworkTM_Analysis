"""Generate sparse binary test arrays and run the localhost UDP/TE test.
No physiological data. Reproduce with python synthetic_benchmark.py.
"""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
from udp_te import benchmark
import csv
from collections import defaultdict


def synthetic(seed, rate, n=40000, channels=128):
    rng=np.random.default_rng(seed)
    a=(rng.random((n,channels)) < rate).astype(np.uint8)
    # Introduce one-lag copy events between adjacent pairs; no physiological model.
    # Copy uses the independent base source; target is OR of background and copy.
    a[1:,1::2] |= (a[:-1,::2] & (rng.random((n-1,channels//2))<0.2))
    return a


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='results_synthetic');p.add_argument('--reps',type=int,default=4);a=p.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    specs=[('synthetic_low.npy',74001,0.0025),('synthetic_high.npy',74002,0.0055)]
    with tempfile.TemporaryDirectory() as tmp:
        paths=[]
        for name,seed,rate in specs:
            path=Path(tmp)/name;np.save(path,synthetic(seed,rate));paths.append(path)
        benchmark(paths,out,a.reps)
    meta=json.loads((out/'metadata.json').read_text())
    for item,(_,seed,rate) in zip(meta,specs):
        item.update(provenance='Synthetic binary events (not neuronal recordings)',generator_seed=seed,base_rate_per_sample=rate,copy_probability=0.2)
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    with (out/'socket_results.csv').open() as f: rows=list(csv.DictReader(f))
    groups=defaultdict(list)
    for r in rows:groups[(float(r['loss']),float(r['persistence']),r['mode'])].append(r)
    summary=[]
    for (loss,persistence,mode),rr in sorted(groups.items()):
        rec=dict(loss=loss,persistence=persistence,mode=mode,runs=len(rr))
        for col in ['received','valid','masked_rmse','zero_rmse','bt','br']:
            rec[col]=float(np.mean([float(r[col]) for r in rr]))
        rec['total_application_bytes']=rec.pop('bt')+rec.pop('br')
        summary.append(rec)
    for r in summary:
        base=next(b['total_application_bytes'] for b in summary if (b['loss'],b['persistence'],b['mode'])==(r['loss'],r['persistence'],'none'))
        r['byte_ratio_to_no_repair']=r['total_application_bytes']/base
    with (out/'socket_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    assert len(rows)==len(specs)*2*3*a.reps and all(r['payload_match']=='True' for r in rows)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
