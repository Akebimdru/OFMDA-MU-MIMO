import json, sys, time
import numpy as np
from mapproxy import run

def B(x0,x1,y0,y1,h,mat="metal"): return dict(x0=x0,x1=x1,y0=y0,y1=y1,h=h,mat=mat)
ROOM=(16.0,8.0,5.0); BS=(8.0,4.0,4.95)
DIAG=lambda s: [s[3],s[2],s[1],s[0]]

old_s=[(0.5,0.5),(0.5,7.5),(15.5,0.5),(15.5,7.5)]
SPECS={
 "old": dict(name="Current map", room=ROOM, bs=BS, starts=old_s, goals=DIAG(old_s),
   boxes=[B(2.75,3.25,1.5,6.5,5,"concrete"),B(9,14,2.75,3.25,5,"concrete"),B(7.6,8.4,0.6,1.4,5,"concrete"),B(8.6,9.4,5.6,6.4,5,"concrete")]),
}
racks=[]
for xc in (3.0,5.5,10.5,13.0):
    racks += [B(xc-0.4,xc+0.4,1.0,3.3,2.5), B(xc-0.4,xc+0.4,4.7,7.0,2.5)]
sA=[(0.75,1.5),(0.75,6.5),(15.25,1.5),(15.25,6.5)]
SPECS["A0"]=dict(name="A0: warehouse racks", room=ROOM, bs=BS, starts=sA, goals=DIAG(sA), boxes=racks)

S4=[(1.0,1.0),(1.0,7.0),(15.0,1.0),(15.0,7.0)]
# A: parallel aisles along x (ULA axis); racks metal 3 m
SPECS["A"]=dict(name="A: parallel aisles", room=ROOM, bs=BS, starts=S4, goals=DIAG(S4),
  boxes=[B(3,7.2,1.8,2.6,3), B(8.8,13,1.8,2.6,3), B(3,7.2,5.4,6.2,3), B(8.8,13,5.4,6.2,3)])
# B: central partition under BS with two gates + two concrete machine blocks
SPECS["B"]=dict(name="B: central gates", room=ROOM, bs=BS, starts=S4, goals=DIAG(S4),
  boxes=[B(7.75,8.25,0,2.0,2.5,"concrete"), B(7.75,8.25,3.2,4.8,2.5,"concrete"), B(7.75,8.25,6.0,8.0,2.5,"concrete"),
         B(3.5,5.0,3.0,5.0,5,"concrete"), B(11.0,12.5,3.0,5.0,5,"concrete")])
# C: machine islands (partial height) making mid-path shadows
SPECS["C"]=dict(name="C: machine islands", room=ROOM, bs=BS, starts=S4, goals=DIAG(S4),
  boxes=[B(4.5,6.5,2.0,3.2,3,"concrete"), B(9.5,11.5,4.8,6.0,3,"concrete"),
         B(4.5,6.5,5.6,6.4,2,"metal"), B(9.5,11.5,1.6,2.4,2,"metal")])


# v2
SPECS["A2"]=dict(name="A: long parallel aisles", room=ROOM, bs=BS, starts=S4, goals=DIAG(S4),
  boxes=[B(2.5,13.5,2.2,2.8,3), B(2.5,13.5,5.2,5.8,3)])
SPECS["B2"]=dict(name="B: central partition with 2 gates", room=ROOM, bs=BS, starts=S4, goals=DIAG(S4),
  boxes=[B(7.75,8.25,0,1.0,2.5,"concrete"), B(7.75,8.25,2.2,5.8,2.5,"concrete"), B(7.75,8.25,7.0,8.0,2.5,"concrete"),
         B(4.0,5.5,1.0,2.6,5,"concrete"), B(10.5,12.0,5.4,7.0,5,"concrete")])
BSW=(8.0,0.3,4.95)
SPECS["C2"]=dict(name="C: wall-mounted BS + islands", room=ROOM, bs=BSW, starts=S4, goals=DIAG(S4),
  boxes=[B(5.0,6.5,2.5,3.5,5,"concrete"), B(9.5,11.0,2.5,3.5,5,"concrete"), B(7.5,8.5,5.5,6.5,2.5,"metal")])

SB=[(1.0,1.0),(15.0,1.0),(1.0,7.0),(15.0,7.0)]   # AMR1 SW, AMR2 SE, AMR3 NW, AMR4 NE ; goals diagonal
SPECS["B3"]=dict(name="B: partition with two gates", room=ROOM, bs=BS, starts=SB, goals=[SB[3],SB[2],SB[1],SB[0]],
  boxes=[B(0,3.5,3.75,4.25,2.5,"concrete"), B(5.0,11.0,3.75,4.25,2.5,"concrete"), B(12.5,16,3.75,4.25,2.5,"concrete")])

part=[B(0,3.5,3.75,4.25,2.5,"concrete"), B(5.0,11.0,3.75,4.25,2.5,"concrete"), B(12.5,16,3.75,4.25,2.5,"concrete")]
FINAL={
 "A": dict(name="A: parallel aisles (warehouse)", room=ROOM, bs=BS, starts=S4, goals=DIAG(S4),
   boxes=[B(2.5,13.5,2.2,2.8,3), B(2.5,13.5,5.2,5.8,3)]),
 "B": dict(name="B: partition with two gates", room=ROOM, bs=BS, starts=SB, goals=[SB[3],SB[2],SB[1],SB[0]],
   boxes=part+[B(6.0,6.5,0.0,2.8,5,"concrete"), B(9.5,10.0,5.2,8.0,5,"concrete")]),
 "C": dict(name="C: wall-mounted BS + islands", room=ROOM, bs=(8.0,0.3,4.95), starts=S4, goals=DIAG(S4),
   boxes=[B(4.0,5.2,1.6,2.6,5,"concrete"), B(10.8,12.0,1.6,2.6,5,"concrete"), B(7.0,9.0,4.5,5.3,2.5,"metal")]),
}
SPECS.update({"F"+k: v for k, v in FINAL.items()})
if __name__=="__main__":
    for k in sys.argv[2:]:
        import mapproxy
        mapproxy.CSI["nmse_db"]=float(sys.argv[1]) if sys.argv[1]!="none" else None
        t=time.time(); r=run(SPECS[k], f"maps/proxy_{k}.png")
        print(k, f"{time.time()-t:.0f}s", json.dumps({a:b for a,b in r.items() if not a.endswith('_rates')}, ensure_ascii=False))
