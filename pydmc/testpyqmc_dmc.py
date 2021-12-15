'''
Testing Ewald + jellium routines considering only KE + Coulomb energies (no phonons)
'''

#!/usr/bin/env python
import numpy as np
import sys
sys.path.append("../")
from metropolis import metropolis_sample
import pandas as pd
import matplotlib.pyplot as plt
from updatedjastrow import UpdatedJastrow, GetEnergy

#define various constants
elec = 1.602E-19*2997924580 #convert C to statC
hbar = 1.054E-34 #J*s
m = 9.11E-31 #kg
w = 0.1*1.602E-19/hbar
epssr = 23000
epsinf = 2.394**2
conv = 1E-9/1.602E-19 #convert statC^2 (expressions with elec^2) to eV
convJ = 1/1.602E-19 #convert J to eV
eta_STO = epsinf/epssr
alpha = (elec**2*1E-9)/hbar*np.sqrt(m/(2*hbar*w))*1/epsinf*(1 - epsinf/epssr) #convert statC to J*m
U_STO = elec**2/(epsinf*hbar)*np.sqrt(2*m/(hbar*w))*1E-9 #convert statC in e^2 term into J to make U dimensionless
Ry = m*elec**4*(1E-9)**2/(2*epsinf**2 *hbar**2)*1/1.602E-19 #Rydberg energy unit in media, eV
a0 = hbar**2*epsinf/(m*elec**2 *1E-9); #Bohr radius in media
l = np.sqrt(hbar/(2*m*w))/ a0 #phonon length in units of the Bohr radius  

#####################################

def acceptance(posold, posnew, driftold, driftnew, tau, wf):
    """
    Acceptance for importance sampling
    Input:
      poscur: electron positions before move (nelec,ndim,nconf) 
      posnew: electron positions after move (nelec,ndim,nconf)
      driftnew: drift vector at posnew 
      tau: time step
      wf: wave function object
      configs: DMC driver configuration
    Return:
      ratio: [backward move prob.]/[forward move prob.]
      """
    #check axes of summation: originally (nelec, ndim, nconfigs)
    #now (nconfigs, nelec, ndim)
    gfratio = np.exp(
        -np.sum((posold - posnew - driftnew) ** 2 / (2 * tau), axis=(1, 2))
        + np.sum((posnew - posold - driftold) ** 2 / (2 * tau), axis=(1, 2))
    )
    
    ratio = wf.val(posnew) ** 2 / wf.val(posold) ** 2
    return np.minimum(1,ratio * gfratio)

def popcontrol(pos, weight, wavg, wtot):
    probability = np.cumsum(weight / wtot)
    randnums = np.random.random(nconfig)
    new_indices = np.searchsorted(probability, randnums)
    posnew = pos[new_indices, :, :]
    weight.fill(wavg)
    return posnew, weight

from itertools import product
def simple_dmc(wf, ham, tau, pos, popstep=1, nstep=1000, L=10):
    """
  Inputs:
  L: box length (units of a0)
  pos: initial position
  nstep: total number of steps in the sim

  Outputs:
  A Pandas dataframe with each 
  """
    df = {
        "step": [],
        "nconfig": [],
        "r_s": [],
        "ke": [],
        "elocal": [],
        "weight": [],
        "weightvar": [],
        "elocalvar": [],
        "eref": [],
        "tau": [],
        "popstep": [],
    }
    nconfig = pos.shape[0]
    weight = np.ones(nconfig)
    #setup wave function
    configs = wf.setup(pos)
    if nconfig != wf.nconfig:
        print("Incompatible number of walkers: sim nconfig = " + str(nconfig) + ", but wf nconfig = " + str(wf.nconfig) + ". Please re-run step1_opt.py for " + str(nconfig) + " walkers, then try again. Exiting program...")
        return
    eloc = GetEnergy(wf,configs,pos,'total')
    eref = np.mean(eloc)
    print(eref)

    for istep in range(nstep):
        driftold = tau * wf.grad(pos)
        elocold = GetEnergy(wf,configs,pos,'total')

        # Drift+diffusion 
        #with importance sampling
        posnew = pos + np.sqrt(tau) * np.random.randn(*pos.shape) + driftold
        driftnew = tau * wf.grad(posnew)
        acc = acceptance(pos, posnew, driftold, driftnew, tau, wf)
        imove = acc > np.random.random(nconfig)
        pos[imove,:, :] = posnew[imove,:, :]
        acc_ratio = np.sum(imove) / nconfig

        ke = GetEnergy(wf,configs,pos,'ke') #syncs internal wf configs object + driver configs object
        eloc = GetEnergy(wf,configs,pos,'total') #syncs internal wf configs object + driver configs object
        
        oldwt = np.mean(weight)
        weight = weight* np.exp(-0.5* tau * (elocold + eloc - 2*eref))
        
        # Branch
        wtot = np.sum(weight)
        wavg = wtot / nconfig
        
        if istep % popstep == 0:
            pos, weight = popcontrol(pos, weight, wavg, wtot)
            wf.update(configs,pos)

        # Update the reference energy
        Delta = -1./tau* np.log(wavg/oldwt) #need to normalize <w_{n+1}>/<w_n>
        eref = eref + Delta

        if istep % popstep == 0:
            print(
                "iteration",
                istep,
                "avg wt",
                wavg.real,
                "average energy",
                np.mean(eloc * weight / wavg),
                "eref",
                eref,
                "sig_gth",
                np.std(eloc),
            )

        df["step"].append(istep)
        df["ke"].append(np.mean(ke))
        df["elocal"].append(np.mean(eloc))
        df["weight"].append(np.mean(weight))
        df["elocalvar"].append(np.std(eloc))
        df["weightvar"].append(np.std(weight))
        df["eref"].append(eref)
        df["tau"].append(tau)
        df["r_s"].append(r_s)
        df["nconfig"].append(nconfig)
        df['popstep'].append(popstep)
    return pd.DataFrame(df)

def simple_vmc(wf, ham, tau, pos, nstep=1000, L=10):
    """
    Force every walker's weight to be 1.0 at every step, and never create/destroy walkers (i.e. no drift, no weights). Uses Metropolis algorithm to accept/reject steps and ensure MC has |psi_T|^2 as its equilibrium distribution.

    In practice, the following two steps should be sufficient for VMC:
    1. keep diffusion term so that electrons move from one step to another R -> R'
    2. use Metropolis criteria to accept/reject according to |Psi_T|^2(R')/|Psi_T|^2(R)
    No weights are needed (a.k.a. set weight=1 for all walkers at every step)

    Inputs:
        L: box length (units of a0)
 
    Outputs:
        A Pandas dataframe with each 

    """
    df = {
        "step": [],
        "r_s": [],
        "tau": [],
        "elocal": [],
        "ke": [],
        "pot": [],
        "acceptance": [],
    }
    nconfig = pos.shape[0]
    weight = np.ones(nconfig)

    _,_,eloc = PBCjell_E(pos, wf, ham)
    eref = np.mean(eloc)
    print(eref)

    blocksize = 1 #units of Bohr radius a0
    nblocks = int(L/blocksize)
    bins = np.linspace(0,L,nblocks)
    print(bins)
    #hist = hist_reblock(pos, bins)
    for istep in range(nstep):
        wfold=wf.value(pos)
        _,_,elocold = PBCjell_E(pos, wf, ham)
        # propose a move
        gauss_move_old = np.random.randn(*pos.shape)
        posnew=pos + np.sqrt(tau)*gauss_move_old

        wfnew=wf.value(posnew)
        # calculate Metropolis-Rosenbluth-Teller acceptance probability
        prob = wfnew**2/wfold**2 # for reversible moves
        # get indices of accepted moves
        acc_idx = (prob + np.random.random_sample(nconfig) > 1.0)
        # update stale stored values for accepted configurations
        pos[:,:,acc_idx] = posnew[:,:,acc_idx]
        wfold[acc_idx] = wfnew[acc_idx]
        acceptance = np.mean(acc_idx) #avg acceptance rate at each step (NOT total, would have to additionally divide by nstep)
        ke,ewald,eloc = PBCjell_E(pos, wf, ham)

        #update histogram of electron positions (e- density)
        #hist = hist + hist_reblock(pos, bins)

        #oldwt = np.mean(weight)
        #weight = weight* np.exp(-0.5* tau * (elocold + eloc - 2*eref))

        if istep % 10 == 0:
            print(
                "iteration",
                istep,
                "ke", np.mean(ke), "ewald", np.mean(ewald),
                "average energy",
                np.mean(eloc),
                "acceptance",acceptance
            )
        #weight.fill(1.)

        df["step"].append(istep)
        df["pot"].append(np.mean(ewald))
        df["ke"].append(np.mean(ke))
        df["elocal"].append(np.mean(eloc))
        df["acceptance"].append(acceptance)
        df["tau"].append(tau)
        df["r_s"].append(r_s)

    return pd.DataFrame(df)

#####################################

if __name__ == "__main__":
    from slaterwf import ExponentSlaterWF
    from updatedjastrow import UpdatedJastrow
    from ham import Hamiltonian
    import time

    tproj = 128 #projection time = tau * nsteps

    nconfig = 512 #default is 5000
    dfs = []
    r_s = int(sys.argv[1]) #inter-electron spacing, controls density
    L = (4*np.pi*2/3)**(1/3) * r_s #sys size/length measured in a0; multiply by 2 since 2 = # of electrons
    print("L",L)
    U = 2.
    csvname = "pyqmc_rs_" + str(r_s) + "_popsize_" + str(nconfig) + ".csv"
    wf = UpdatedJastrow(r_s)
    ham = Hamiltonian(U=U, L=L)

    np.random.seed(0)
    tic = time.perf_counter()
     
    for tau in [r_s/20, r_s/40, r_s/80]:
    #for tau in [r_s/20]:
        nstep = int(tproj/tau)
        print(nstep)
        
        dfs.append(
            simple_dmc(
                wf,
                ham,
                pos= L* np.random.rand(nconfig, 2, 3), 
                L=L,
                tau=tau,
                popstep=10,
                nstep=nstep #orig: 10000
            )
        )
    csvname = 'DMC_' + csvname
    
    ''' 
    for tau in [r_s/10,r_s/20, r_s/40,r_s/80]:
        nstep = int(tproj/tau)
        print(nstep)
        dfs.append(
            simple_vmc(
                wf,
                ham,
                pos= L* np.random.rand(2, 3, nconfig), 
                L=L,
                tau=tau,
                nstep=nstep #orig: 10000
            )
        )
    csvname = 'VMC_' + csvname
    '''
    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")

    df = pd.concat(dfs)
    df.to_csv(csvname, index=False)
     
