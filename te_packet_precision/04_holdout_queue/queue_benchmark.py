"""Event-driven abstract shared-link FIFO queue; NOT a physical network measurement.
Each packet (including control) is one equal-sized unit. Original packets arrive every
10 ms, four per group. One XOR parity or at most one NACK+retry per group. A fixed
finite buffer and capacity are shared by both directions; random post-service erasures.
Timing is illustrative; no actual packet bytes, TCP/RTP or congestion-control stack.
"""
import heapq, collections, itertools, csv
from pathlib import Path
import numpy as np
import validate as v

def schedule(seed, n, service, p, mode, capacity=8, prop=8):
    rng=np.random.default_rng(seed); seq=itertools.count(); events=[]; departures=collections.deque(); arrivals=np.full(n,np.inf); parity=np.full(n//4,np.inf); sent=drop=erased=0; waits=[]
    def add(t,kind,j):heapq.heappush(events,(t,next(seq),kind,j))
    for j in range(n):add(j*10.,'data',j)
    for g in range(n//4):
        if mode=='xor':add((4*g+3)*10.+.001,'parity',g)
        elif mode=='nack':add((4*g+3)*10.+20.,'check',g)
    while events:
        t,_,kind,j=heapq.heappop(events)
        if kind=='arrival':arrivals[j]=min(arrivals[j],t);continue
        if kind=='parity_arrival':parity[j]=t;continue
        if kind=='feedback_arrival':add(t,'retry',j);continue
        if kind=='check':
            missed=np.flatnonzero(arrivals[4*j:4*j+4]>t)
            if len(missed):add(t,'feedback',4*j+int(missed[0]))
            continue
        sent+=1
        # one independent erasure draw per offered transmission, including dropped packets
        lost=rng.random()<p
        while departures and departures[0]<=t:departures.popleft()
        if len(departures)>=capacity:drop+=1;continue
        start=max(t,departures[-1] if departures else t);end=start+service;departures.append(end);waits.append(start-t)
        if lost:erased+=1;continue
        event={'data':'arrival','retry':'arrival','parity':'parity_arrival','feedback':'feedback_arrival'}[kind]
        add(end+prop,event,j)
    if mode=='xor':
        for g in range(n//4):
            a=arrivals[4*g:4*g+4].copy()
            for i in range(4):
                arrivals[4*g+i]=min(a[i],max(parity[g],np.max(np.delete(a,i))))
    return arrivals,sent/n,drop/sent,erased/sent,float(np.mean(waits))

def main():
    n=v.N//10; reps=40; c=v.signal(77120,.98,reps); full=v.te(c,np.ones_like(c,bool)); rows=[]
    for service,p in itertools.product((6.,9.,12.),(.1,.3)):
        for mode in ('none','xor','nack'):
            arr=[]; rates=[]; drops=[]; erasures=[]; waits=[]
            for i in range(reps):
                a,rate,drop,erase,wait=schedule(77200+i,n,service,p,mode);arr.append(a);rates.append(rate);drops.append(drop);erasures.append(erase);waits.append(wait)
            ar=np.array(arr).T
            for D in (50,100):
                receive=ar<=np.arange(n)[:,None]*10+D;m=np.repeat(receive,10,axis=0);w=m[:-1]&m[1:];e=v.te(c,w)
                rows.append(dict(service_ms=service,offered_data_utilization=service/10,erasure_probability=p,deadline_ms=D,mode=mode,reps=reps,packet_budget=1.5,mean_offered_packet_ratio=np.mean(rates),queue_drop_fraction=np.mean(drops),post_service_erasure_fraction=np.mean(erasures),mean_queue_wait_ms=np.mean(waits),deadline_delivery=receive.mean(),valid_fraction=w.mean(),te_rmse_vs_complete=np.sqrt(np.mean((e-full)**2)),te_variance=e.var(ddof=1),complete_te_variance=full.var(ddof=1)))
        print('done',service,p,flush=True)
    with open(Path(__file__).parent/'queue_summary.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
