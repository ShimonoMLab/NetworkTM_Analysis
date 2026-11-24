"""One-host UDP payload verification; logical timing/loss, no wall-clock network claims."""
import socket, struct, numpy as np, csv
from experiment import signals,masks,N,P,OUT

def verify(kind,mode,D,seed=11000):
 x,y=signals(1000); inp=np.packbits(np.concatenate([x,y],axis=1),axis=1).reshape(-1,200); b=N//40
 r=np.random.default_rng(seed);a=r.random(b*6);z=a<.1
 if kind=='burst':
  for k in range(1,len(z)):z[k]=a[k]<(.65 if z[k-1] else .1*.35/.9)
 z=z.reshape(b,6);out=np.zeros_like(inp);seen=np.zeros(len(inp),bool);byte_count=0;maxblock=0
 with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as A,socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as B:
  for s in [A,B]:s.bind(('127.0.0.1',0));s.settimeout(1)
  def send(src,dst,payload,k,slot):
   nonlocal byte_count
   pkt=struct.pack('!II',k,slot)+bytes(payload);src.sendto(pkt,dst.getsockname());got,_=dst.recvfrom(4096);assert got==pkt;byte_count+=len(pkt)+28 # IPv4+UDP headers, no link headers
   return None if z[k,slot] else np.frombuffer(got[8:],np.uint8).copy()
  for k in range(b):
   start=byte_count;data=inp[k*4:k*4+4];rx=[send(A,B,p,k,i) for i,p in enumerate(data)];t=[i*10+20 if p is not None else 999 for i,p in enumerate(rx)];miss=[i for i,p in enumerate(rx) if p is None]
   if mode=='xor':
    par=send(A,B,np.bitwise_xor.reduce(data,axis=0),k,4)
    if par is not None and len(miss)==1:
     j=miss[0];rx[j]=par.copy()
     for i,p in enumerate(rx):
      if i!=j:rx[j]^=p
     t[j]=60
   if mode=='nack' and miss:
    j=miss[0];fb=np.zeros(200,np.uint8);fb[0]=j;ack=send(B,A,fb,k,4)
    if ack is not None:
     rx[j]=send(A,B,data[int(ack[0])],k,5)
     if rx[j] is not None:t[j]=90
   for i,p in enumerate(rx):
    if p is not None and t[i]<=i*10+D:out[4*k+i]=p;seen[4*k+i]=True
   maxblock=max(maxblock,byte_count-start)
 expected,rate=masks(seed,kind,mode,D);assert np.array_equal(np.repeat(seen,10),expected);assert np.array_equal(out[seen],inp[seen]);assert maxblock<=6*236
 return [kind,mode,D,seen.sum(),byte_count,maxblock,rate,'PASS']
if __name__=='__main__':
 with open(OUT/'loopback.csv','w') as f:
  w=csv.writer(f);w.writerow(['loss','mode','deadline_ms','valid_packets','total_ip_udp_app_bytes','max_bytes_per_block','rate','payload_match'])
  for kind in ['iid','burst']:
   for mode in ['none','xor','nack']:
    for D in [50,100]:w.writerow(verify(kind,mode,D))
 print('12 loopback conditions passed')
