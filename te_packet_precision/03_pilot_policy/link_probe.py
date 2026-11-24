"""Two-process UDP RTT/payload probe; does not measure synchronized one-way delay or TE."""
import argparse, socket, struct, time, csv, statistics

def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['server','client']);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=49069);p.add_argument('--count',type=int,default=100);p.add_argument('--interval',type=float,default=.02);p.add_argument('--output',default='link_probe.csv');a=p.parse_args();assert 1<=a.count<=10000 and a.interval>=0.01
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as s:
        s.settimeout(3 if a.role=='server' else .5)
        if a.role=='server':
            s.bind((a.host,a.port));n=0
            while n<a.count:
                try:d,addr=s.recvfrom(4096)
                except socket.timeout:break
                if len(d)!=208:continue
                s.sendto(d,addr);n+=1
            print('echoed',n);return
        rows=[]
        for i in range(a.count):
            b=struct.pack('!Q',i)+bytes((i+j)%256 for j in range(200));t=time.monotonic_ns();s.sendto(b,(a.host,a.port));ok=False;rtt=None
            while (time.monotonic_ns()-t)/1e9<.5:
                try:d,_=s.recvfrom(4096)
                except socket.timeout:break
                if d==b:ok=True;rtt=(time.monotonic_ns()-t)/1e6;break
            rows.append((i,ok,rtt));time.sleep(a.interval)
        with open(a.output,'w',newline='') as f:w=csv.writer(f);w.writerow(['sequence','payload_echo_match','rtt_ms']);w.writerows(rows)
        vals=[r[2] for r in rows if r[1]];print('matches',len(vals),'/',a.count,'median RTT ms',statistics.median(vals) if vals else None)
if __name__=='__main__':main()
