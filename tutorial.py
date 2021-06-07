#!/usr/bin/env python
import numpy as np
import sys

sys.path.append("../StochasticSchool/Day1/VMC/solutions")

from metropolis import metropolis_sample
import pandas as pd

#####################################


def ke_pot_tot_energies(pos, wf, ham):
    """ calculate kinetic, potential, and local energies
    Input:
      pos: electron positions (nelec,ndim,nconf) 
      wf: wavefunction
      ham: hamiltonian
    Return:
      ke: kinetic energy
      pot: potential energy
      eloc: local energy
    """
    ke = -0.5 * np.sum(wf.laplacian(pos), axis=0)
    pot = ham.pot(pos)
    eloc = ke + pot
    return ke, pot, ke+pot


#####################################


def acceptance(posold, posnew, driftold, driftnew, tau, wf):
    """Input:
      poscur: electron positions before move (nelec,ndim,nconf) 
      posnew: electron positions after  move (nelec,ndim,nconf)
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
    return ratio * gfratio


def simple_dmc(wf, ham, tau, pos, nstep=1000):
    """
  Inputs:
  
  Outputs:
  A Pandas dataframe with each 

  """
    df = {
        "step": [],
        "elocal": [],
        "weight": [],
        "weightvar": [],
        "elocalvar": [],
        "eref": [],
        "tau": [],
    }
    nconfig = pos.shape[2]
    pos, acc = metropolis_sample(pos, wf, tau=0.5) #what's the point of this line? Isn't it already calculating the final position after 1000 steps? Then why bother with the other nstep loop? (i.e. why not just keep pos as the input position matrix)
    weight = np.ones(nconfig)
    ke, pot, eloc = ke_pot_tot_energies(pos, wf, ham)
    eref = np.mean(eloc)

    for istep in range(nstep):
        # Drift+diffusion
        #driftold = tau * wf.gradient(pos)
        ke, pot, elocold = ke_pot_tot_energies(pos, wf, ham)
        pos = pos + np.sqrt(tau) * np.random.randn(*pos.shape)

        '''
        posnew = pos + np.sqrt(tau) * np.random.randn(*pos.shape) + driftold
        driftnew = tau * wf.gradient(posnew)
        acc = acceptance(pos, posnew, driftold, driftnew, tau, wf)
        imove = acc > np.random.random(nconfig)
        pos[:, :, imove] = posnew[:, :, imove]
        acc_ratio = np.sum(imove) / nconfig
        acc_ratio=1
        '''

        # Change weight
        ke, pot, eloc = ke_pot_tot_energies(pos, wf, ham)
        oldwt = weight
        weight *= np.exp(-0.5 * tau * (eloc + elocold - 2 * eref))

        # Branch
        wtot = np.sum(weight)
        wavg = wtot / nconfig
        
        probability = np.cumsum(weight / wtot)
        randnums = np.random.random(nconfig)
        new_indices = np.searchsorted(probability, randnums)
        pos = pos[:, :, new_indices]
        
        # Update the reference energy
        Delta = -1./tau* np.log(wavg/np.mean(oldwt)) #need to normalize <w_{n+1}>/<w_n>
        E_gth = eref + Delta
        eref = eref + Delta

        print(
            "iteration",
            istep,
            "avg wt",
            wavg,
            "average energy",
            np.mean(eloc * weight / wavg),
            "eref",
            eref,
            "E_gth",
            E_gth,
            "sig_gth",
            np.std(eloc),
            #"acceptance",
            #acc_ratio,
        )
        df["step"].append(istep)
        df["elocal"].append(np.mean(eloc))
        df["weight"].append(np.mean(weight))
        df["elocalvar"].append(np.std(eloc))
        df["weightvar"].append(np.std(weight))
        df["eref"].append(eref)
        df["tau"].append(tau)
        #This is part of branching but we accumulate before this so you can see the weights.
        weight.fill(wavg)

    return pd.DataFrame(df)


#####################################

if __name__ == "__main__":
    from slaterwf import ExponentSlaterWF
    from wavefunction import MultiplyWF, JastrowWF
    from hamiltonian import Hamiltonian

    nconfig = 15 #5000
    dfs = []
    for tau in [0.01]: #,0.005, 0.0025]:
        dfs.append(
            simple_dmc(
                JastrowWF(0.5),
                #MultiplyWF(ExponentSlaterWF(2.0), JastrowWF(0.5)),
                Hamiltonian(),
                pos=np.random.randn(2, 3, nconfig),
                tau=tau,
                nstep=1000, #10000
            )
        )
    df = pd.concat(dfs)
    df.to_csv("tutorial.csv", index=False)
