import pandas as pd
import numpy as np
import sys

def reblock(eloc,warmup,nblocks):
    elocblock=np.array_split(eloc[warmup:],nblocks) #split local energy array into nblocks subarrays
    print(tau,len(elocblock))
    blockenergy=[np.mean(x) for x in elocblock]
    return np.mean(blockenergy),np.std(blockenergy)/np.sqrt(nblocks)

warmup=50
blocksize=1.0 # in Hartree-1

df=pd.read_csv(sys.argv[1])
dfreblock=[]
for tau,grp in df.groupby("tau"):
    blocktau=blocksize/tau
    eloc=grp.sort_values('step')['elocal'].values
    nblocks=int((len(eloc)-warmup)/blocktau)
    avg,err=reblock(eloc,warmup,nblocks)
    dfreblock.append({'tau':tau,
        'eavg':avg,
        'err':err})

pd.DataFrame(dfreblock).to_csv("jellium_rs_2_reblocked.csv")


