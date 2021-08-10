'''
Testing Ewald + jellium routines considering only KE + Coulomb energies (no phonons)
'''

#!/usr/bin/env python
import numpy as np
import sys

from metropolis import metropolis_sample
import pandas as pd

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

def jellium_E(pos, wf, ham):
    '''returns kinetic energy, Ewald energy (e-e only), and total potential in Rydbergs'''

    ke = -np.sum(wf.laplacian(pos), axis=0) #should have coeff of -1 for actual calc
    return ke, ham.ewald(pos), ke + ham.ewald(pos)

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
    Return:
      ratio: [backward move prob.]/[forward move prob.]
      """
    gfratio = np.exp(
        -np.sum((posold - posnew - driftnew) ** 2 / (2 * tau), axis=(0, 1))
        + np.sum((posnew - posold - driftold) ** 2 / (2 * tau), axis=(0, 1))
    )
    ratio = wf.value(posnew) ** 2 / wf.value(posold) ** 2
    return np.minimum(1,ratio * gfratio)

def popcontrol(pos, weight, wavg, wtot):
    probability = np.cumsum(weight / wtot)
    randnums = np.random.random(nconfig)
    new_indices = np.searchsorted(probability, randnums)
    posnew = pos[:, :, new_indices]
    weight.fill(wavg)
    return posnew, weight

def periodic(pos, L):
    return None

from itertools import product
def simple_dmc(wf, ham, tau, pos, popstep=1, nstep=1000, L=10):
    """
  Inputs:
  L: box length (units of a0)
 
  Outputs:
  A Pandas dataframe with each 

  """
    df = {
        "step": [],
        "r_s": [],
        "elocal": [],
        "weight": [],
        "weightvar": [],
        "elocalvar": [],
        "eref": [],
        "tau": [],
        "popstep": [],
    }
    nconfig = pos.shape[2]
    weight = np.ones(nconfig)

    _,_,eloc = jellium_E(pos, wf, ham)
    eref = np.mean(eloc)
    print(eref)

    for istep in range(nstep):
        rdist = np.mean(np.sum((pos[0,:,:]-pos[1,:,:])**2,axis=0)**0.5)
        
        driftold = tau * wf.gradient(pos)
        _,_,elocold = jellium_E(pos, wf, ham)

        # Drift+diffusion 
        #with importance sampling
        posnew = pos + np.sqrt(tau) * np.random.randn(*pos.shape) + driftold
        driftnew = tau * wf.gradient(posnew)
        acc = acceptance(pos, posnew, driftold, driftnew, tau, wf)
        imove = acc > np.random.random(nconfig)
        pos[:, :, imove] = posnew[:, :, imove]
        acc_ratio = np.sum(imove) / nconfig

        #impose periodic boundary conditions - THIS IS WRONG, NEED TO IMPLEMENT MINIMUM IMAGE CONVENTION
        #pos = pos % L
        
        #eloc, _, _, rho, f2p = eph_energies(pos, wf, ham, tau, h_ks, f_ks, ks, kcopy)
        ke,ewald,eloc = jellium_E(pos, wf, ham)
        
        oldwt = np.mean(weight)
        weight = weight* np.exp(-0.5* tau * (elocold + eloc - 2*eref))
        
        # Branch
        wtot = np.sum(weight)
        wavg = wtot / nconfig
        
        if istep % popstep == 0:
            pos, weight = popcontrol(pos, weight, wavg, wtot)

        # Update the reference energy
        Delta = -1./tau* np.log(wavg/oldwt) #need to normalize <w_{n+1}>/<w_n>
        eref = eref + Delta

        if istep % popstep == 0:
            print(
                "iteration",
                istep,
                "sep dist",
                rdist,
                #"ke", np.mean(ke), "ewald", np.mean(ewald),
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
        df["elocal"].append(np.mean(eloc))
        df["weight"].append(np.mean(weight))
        df["elocalvar"].append(np.std(eloc))
        df["weightvar"].append(np.std(weight))
        df["eref"].append(eref)
        df["tau"].append(tau)
        df["r_s"].append(r_s)
        df['popstep'].append(popstep)
    return pd.DataFrame(df)

def simple_vmc(wf, ham, tau, pos, nstep=1000, N=5, L=10):
    """
  Force every walker's weight to be 1.0 at every step, and never create/destroy walkers (i.e. no branching, no importance sampling)

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
        "weight": [],
    }
    nconfig = pos.shape[2]
    weight = np.ones(nconfig)
    acceptance = 0.0
    for istep in range(nstep):
        wfold=wf.value(pos)
        _,_,elocold = jellium_E(pos, wf, ham)
        # propose a move
        gauss_move_old = np.random.randn(*pos.shape)
        posnew=pos + np.sqrt(tau)*gauss_move_old
        posnew = posnew % L

        wfnew=wf.value(posnew)

        # calculate Metropolis-Rosenbluth-Teller acceptance probability
        prob = wfnew**2/wfold**2 # for reversible moves

        # get indices of accepted moves
        acc_idx = (prob + np.random.random_sample(nconfig) > 1.0)

        # update stale stored values for accepted configurations
        pos[:,:,acc_idx] = posnew[:,:,acc_idx]
        wfold[acc_idx] = wfnew[acc_idx]
        #acceptance += np.mean(acc_idx)/nstep
        ke,ewald,eloc = jellium_E(pos, wf, ham)

        if istep % 10 == 0:
            print(
                "iteration",
                istep,
                #"ke", np.mean(ke), "ewald", np.mean(ewald),
                "average energy",
                np.mean(eloc),
            )

        df["step"].append(istep)
        df["elocal"].append(np.mean(eloc))
        df["weight"].append(np.mean(weight))
        df["tau"].append(tau)
        df["r_s"].append(r_s)
    return pd.DataFrame(df)

#####################################

if __name__ == "__main__":
    from slaterwf import ExponentSlaterWF
    from wavefunction import MultiplyWF, JastrowWF, UniformWF
    from ham import Hamiltonian
    import time

    tproj = 500 #projection time = tau * nsteps
    tequil = 100 #equilibration time = tau*(# steps thrown out)

    nconfig = 500 #default is 5000
    dfs = []
    r_s = int(sys.argv[1]) #inter-electron spacing, controls density
    L = (4*np.pi*2/3)**(1/3) * r_s #sys size/length measured in a0; multiply by 2 since 2 = # of electrons
    U = 2.
    np.random.seed(0)
    tic = time.perf_counter()
    print("jellium_rs_" + str(r_s) + ".csv")

    for tau in [r_s/10, r_s/20, r_s/40, r_s/80]: #[0.01, 0.005, 0.0025]:
        nstep = int(tproj/tau)
        print(nstep)
        dfs.append(
            simple_dmc(
                UniformWF(),
                #JastrowWF(0.5),
                #MultiplyWF(ExponentSlaterWF(2.0), JastrowWF(0.5)),
                Hamiltonian(U=U, L=L),
                pos= L* np.random.rand(2, 3, nconfig), 
                L=L,
                tau=tau,
                popstep=10,
                nstep=nstep #orig: 10000
            )
        )
        
    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")

    df = pd.concat(dfs)
    df.to_csv("DMCjellium_rs_" + str(r_s) + "_popsize_" + str(nconfig) + "_tproj_" + str(tproj) +  ".csv", index=False)
