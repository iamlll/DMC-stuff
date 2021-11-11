import pandas as pd
import numpy as np
import sys

def reblock(eloc,warmup,nblocks):
    elocblock=np.array_split(eloc[warmup:],nblocks) #throw out "warmup" number of equilibration steps and split resulting local energy array into nblocks subarrays
    print(tau,len(elocblock))
    blockenergy=[np.mean(x) for x in elocblock]
    return np.mean(blockenergy),np.std(blockenergy)/np.sqrt(nblocks)

#allow multiple input files and concatenate results in same output file
filenames = sys.argv[1:]

warmup=1000
tequil = 20 #equilibration time = timestep * (# steps thrown out)
blocksize=1.0 # in Hartree-1

dfreblock=[]

for name in filenames:
    df=pd.read_csv(name)

    for tau,grp in df.groupby("tau"):
        r_s = grp['r_s'].values[0]
        blocktau=blocksize/tau
        eloc=grp.sort_values('step')['ke'].values
        nequil = int(tequil/tau)
        nblocks=int((len(eloc)-nequil)/blocktau)
        avg,err=reblock(eloc,nequil,nblocks)
        dfreblock.append({ 
            'n_equil': nequil,
            'r_s':r_s,
            'tau':tau,
            'eavg':avg/2, #Rydbergs PER ELECTRON, or total hartrees
            'err':err})

pd.DataFrame(dfreblock).to_csv("PBC_reblocked_tequil_" + str(tequil) + ".csv")
