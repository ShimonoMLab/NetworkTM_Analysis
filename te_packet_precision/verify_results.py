"""Check archived numerical claims and optionally reproduce UDP trials. Run from any directory."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, tempfile
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(); p.add_argument('--rerun-udp', action='store_true'); a=p.parse_args()
    def read(path): return pd.read_csv(ROOT/path)
    def near(x,y,atol=0.005): assert abs(x-y)<atol,(x,y)
    for line in (ROOT/'SHA256SUMS.txt').read_text().splitlines():
        h, name=line.split('  ',1); assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    d=read('03_pilot_policy/synthetic_variance.csv'); assert len(d)==72
    near(d.error_fraction.mean()*100,21.66); near(d.error_hac.mean()*100,26.56)
    for path, expected in [('04_holdout_queue/holdout_lag.csv',{'fraction':17.06,'hac':10.07}),('04_holdout_queue/direct_hac_holdout.csv',{'fraction':12.64,'factorized':15.45,'direct_masked':15.51})]:
        d=read(path)
        for k,v in expected.items():
            x=d[d.model==k]; assert len(x)==72; near(x.relative_error.mean()*100,v)
    d=read('05_state_dependent_loss/informative_summary.csv').set_index(['scenario','method']); near(d.loc[('observed_state','complete_case'),'rmse'],.0468844,1e-6); near(d.loc[('observed_state','state_ipw'),'rmse'],.00418863,1e-6); near(d.loc[('unobserved_activity','complete_case'),'rmse'],.04142056,1e-6); near(d.loc[('unobserved_activity','state_ipw'),'rmse'],.00826403,1e-6); assert d.loc[('unobserved_activity','state_ipw'),'bias']<-.007
    d=read('06_udp_integration/socket_results.csv'); assert len(d)==48 and d.payload_match.all(); assert set(d.file)=={'synthetic_low.npy','synthetic_high.npy'}
    meta=json.loads((ROOT/'06_udp_integration/metadata.json').read_text()); assert all('Synthetic' in x['provenance'] for x in meta)
    n=read('07_null_calibration/null_power.csv'); assert len(n)==128
    for est,cal,v in [('mask','complete',.13),('zero','complete',.845),('mask','matched',.065)]:
        x=n[(n['T']==12000)&(n.effect==0)&(n.scenario=='burst30')&(n.estimator==est)&(n.calibration==cal)]; assert len(x)==1; near(x.rate.iloc[0],v,1e-8)
    if a.rerun_udp:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable,str(ROOT/'06_udp_integration/synthetic_benchmark.py'),'--out',tmp,'--reps','4'],check=True,cwd=ROOT/'06_udp_integration',stdout=subprocess.DEVNULL)
            got=pd.read_csv(Path(tmp)/'socket_results.csv'); cols=[c for c in d.columns if c!='elapsed_s']; pd.testing.assert_frame_equal(d[cols],got[cols],check_exact=True); pd.testing.assert_frame_equal(read('06_udp_integration/socket_summary.csv'),pd.read_csv(Path(tmp)/'socket_summary.csv'),check_exact=True); assert meta==json.loads((Path(tmp)/'metadata.json').read_text())
    print('Archived aggregate/integrity checks PASS'+('; 48 fresh UDP trials PASS (all deterministic fields, metadata and summary)' if a.rerun_udp else ''))
if __name__=='__main__':main()
