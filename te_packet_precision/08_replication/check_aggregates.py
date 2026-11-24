"""Independent CSV verification using only the Python standard library."""
from pathlib import Path
import csv,math,statistics,collections,json
R=Path(__file__).resolve().parents[1]
def load(p):
    with (R/p).open() as f:return list(csv.DictReader(f))
def check(x,y):assert math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-12),(x,y)
def grouped(rows,keys):
    out=collections.defaultdict(list)
    for r in rows:out[tuple(r[k] for k in keys)].append(r)
    return out
s=load('05_state_dependent_loss/informative_summary.csv'); rr=grouped(load('05_state_dependent_loss/informative_replicates.csv'),['scenario','method'])
for r in s:
    v=rr[(r['scenario'],r['method'])]; assert len(v)==int(r['n']);err=[float(x['estimate'])-float(x['full_te']) for x in v];check(statistics.mean(err),float(r['bias']));check(math.sqrt(statistics.mean(e*e for e in err)),float(r['rmse']))
s=load('06_udp_integration/socket_summary.csv'); rr=grouped(load('06_udp_integration/socket_results.csv'),['loss','persistence','mode']);rr={ (float(k[0]),float(k[1]),k[2]):v for k,v in rr.items()}
for r in s:
    k=(float(r['loss']),float(r['persistence']),r['mode']);v=rr[k];assert len(v)==int(r['runs']);assert all(x['payload_match']=='True' for x in v)
    for c in ['received','valid','masked_rmse','zero_rmse']:check(statistics.mean(float(x[c]) for x in v),float(r[c]))
    b=statistics.mean(int(x['bt'])+int(x['br']) for x in v);base=statistics.mean(int(x['bt'])+int(x['br']) for x in rr[k[:2]+('none',)]);check(b,float(r['total_application_bytes']));check(b/base,float(r['byte_ratio_to_no_repair']))
n=load('07_null_calibration/null_power.csv');assert len(n)==128
for r in n:check(int(r['reject'])/int(r['n_test']),float(r['rate']))
print('PASS: 6 IPW bias/RMSE rows, 6 UDP summary rows, 128 null-calibration proportions; independent stdlib implementation.')
