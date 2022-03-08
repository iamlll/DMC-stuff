import pandas as pd
import numpy as np
import sys
import os

def reblock(eloc,warmup,nblocks):
    elocblock=np.array_split(eloc[warmup:],nblocks) #throw out "warmup" number of equilibration steps and split resulting local energy array into nblocks subarrays
    blockenergy=[np.mean(x.real) for x in elocblock]
    return np.mean(blockenergy),np.std(blockenergy)/np.sqrt(nblocks)

#allow multiple input files and concatenate results in same output file
filenames = sys.argv[1:]

#warmup=1000
# equilibration time = timestep * (# steps thrown out)
tequil = 20 
blocksize=1.0 # in Hartree energy units

dfreblock=[]

for name in filenames:
    extension = os.path.splitext(name)[1]
    print(extension)
    if 'pkl' in extension:
        df=pd.read_pickle(name)
    else:
        df=pd.read_csv(name)
    for tau,grp in df.groupby("tau"):
        r_s = grp['r_s'].values[0]
        N = grp['N_cut'].values[0]
        if 'alpha' in grp:
            alpha = grp['alpha'].values[0]
        else:
            alpha = (1-grp['eta'].values[0])*grp['l'].values[0]
        blocktau= max(blocksize/tau,blocksize)
        eloc=grp.sort_values('step')['elocal'].values #mixed estimator
        egth=grp.sort_values('step')['egth'].values #growth estimator
        nequil = int(tequil/tau)
        nblocks=int((len(eloc)-nequil)/blocktau)
        avg,err=reblock(eloc,nequil,nblocks)
        avg_gth,err_gth=reblock(egth,nequil,nblocks)
        print(tau,nblocks)
        dfreblock.append({ 
            'n_equil': nequil,
            'alpha': alpha,
            'r_s':r_s,
            'tau':tau,
            'Ncut':N,
            'eavg':avg, #in hartrees, I believe
            'err':err,
            'egth':avg_gth,
            'err_gth':err_gth,
            })

pd.DataFrame(dfreblock).to_csv("pyQMC_reblocked_tequil_" + str(tequil) + ".csv")
