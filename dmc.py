#!/usr/bin/env python
import numpy as np
import sys

#sys.path.append("../../Day1/VMC/solutions")

from metropolis import metropolis_sample
import pandas as pd

#define various constants
elec = 1.602E-19*2997924580 #convert C to statC
hbar = 1.054E-34 #J*s
m = 9.11E-31 #kg
w = 0.1*1.602E-19/hbar
epssr = 23000
epsinf = 2.394**2
conv = 1E-9/1.602E-19 #convert statC (expressions with elec^2) to eV
convJ = 1/1.602E-19 #convert J to eV
eta_STO = epsinf/epssr
alpha = (elec**2*1E-9)/hbar*np.sqrt(m/(2*hbar*w))*1/epsinf*(1 - epsinf/epssr) #convert statC to J*m
l = np.sqrt(hbar/(2*m*w))   
U_STO = elec**2/(epsinf*hbar)*np.sqrt(2*m/(hbar*w))*1E-9 #convert statC in e^2 term into J to make U dimensionless

#####################################

def elec_energies(pos,wf,ham):
    """ calculate kinetic + Coulomb energies
    Input:
      pos: electron positions (nelec,ndim,nconf) 
      wf: wavefunction
      ham: hamiltonian
    Return:
      ke: kinetic energy
      pot: Coulomb energy
    """
    ke = - np.sum(wf.laplacian(pos), axis=0)
    pot = ham.pot_ee(pos)
    return ke, pot

def phonon_energies(pos,wf,ham):
    """ calculate kinetic + Coulomb energies
    Input:
      pos: electron positions (nelec,ndim,nconf) 
      wf: wavefunction
      ham: hamiltonian
    Return:
      ke: kinetic energy
      pot: potential energy
      eloc: local energy
    """
    ke = - np.sum(wf.laplacian(pos), axis=0)
    pot = ham.pot_ee(pos)
    return ke, pot

def ke_pot_tot_energies(pos, wf, ham, f_ks):
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
    ke = - np.sum(wf.laplacian(pos), axis=0)
    pot = ham.pot(pos, f_ks)
    eloc = ke + pot
    return ke, pot, eloc


#####################################


def acceptance(posold, posnew, driftold, driftnew, tau, wf):
    """Input:
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
    #print((ratio[0],gfratio[0]))
    return np.minimum(1,ratio * gfratio)

def init_f_k(ks, g, nconfig):
    '''
    Initialize the phonon displacement functions f_k from the optimized Gaussian result
    input:
        ks: allowed k-vectors in the supercell
    '''
    kmag = np.sum(ks**2,axis=1)**0.5 #find k magnitudes |k|
    #find f_ks
    yopt = 1.39
    sopt = 1.05E-9
    dopt = yopt*sopt #assume pointing in z direction
    f_ks = -2j*g/kmag* np.exp(-kmag**2 * sopt**2/4) * (np.cos(ks[:,2] * yopt*sopt/2) - np.exp(-yopt**2/2) )/(1- np.exp(-yopt**2/2)) #assume d = mu1 - mu2 pointing in the z direction only.
    f_kcopy = np.array([[ f_ks[i] for j in range(nconfig)] for i in range(len(ks))]) #make f_ks array size (# ks) x (# configurations)
    return f_kcopy

def Update_f_k(f_ks,ks,pos, tau, g, kmags):
    #find k dot products with the positions
    dprod1 = np.matmul(ks,pos[0,:,:]) #np array for each k value; k dot r1
    dprod2 = np.matmul(ks,pos[1,:,:]) #k dot r2 
    #eikr1 = np.exp(1j*dprod1) + np.exp(1j*dprod2)
    eikr2 = np.exp(-1j*dprod1) + np.exp(-1j*dprod2)
    return f_ks + tau*1j*g/kmags * eikr2

from itertools import product
def simple_dmc(wf, ham, tau, pos, g, nstep=1000, N=5, L=10):
    """
  Inputs:
  g: DOS for el-ph interaction
  N: number of allowed k-vals in each direction
  L: box size in 1D (idk units)
 
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
    pos, acc = metropolis_sample(pos, wf, tau=0.5)
    weight = np.ones(nconfig)
    #Make a supercell/box
    #k = (nx, ny, nz)*2*pi/L for nx^2+ny^2+nz^2 <= n_c^2 for cutoff value n_c = N, where n_c -> inf is the continuum limit. 
    #A k-sphere cutoff is conventional as it specifies a unique KE cutoff
    ks = 2*np.pi/L* np.array([[nx,ny,nz] for nx,ny,nz in product(range(1,N+1), range(1,N+1), range(1,N+1)) if nx**2+ny**2+nz**2 <= N**2 ])

    kmag = np.sum(ks**2,axis=1)**0.5 #find k magnitudes
    kcopy = np.array([[ kmag[i] for j in range(nconfig)] for i in range(len(kmag))])
    #initialize f_ks
    f_ks = init_f_k(ks,g,nconfig)

    ke, pot, eloc = ke_pot_tot_energies(pos, wf, ham, f_ks)
    eref = np.mean(eloc)

    for istep in range(nstep):
        # Drift+diffusion
        driftold = tau * wf.gradient(pos)
        ke, pot, elocold = ke_pot_tot_energies(pos, wf, ham, f_ks)

        posnew = pos + np.sqrt(tau) * np.random.randn(*pos.shape) + driftold
        driftnew = tau * wf.gradient(posnew)
        acc = acceptance(pos, posnew, driftold, driftnew, tau, wf)
        # get indices of accepted moves
        acc_idx = (acc > np.random.random_sample(nconfig)) #boolean array with len = nconfig = # samples, of form acc_idx = [True = 1, False = 0, ...]

        # update stale stored values for accepted configurations
        pos[:,:,acc_idx] = posnew[:,:,acc_idx]
        acc_ratio = np.mean(acc_idx) #fraction of samples that have been accepted

        #update f_ks
        newf_ks = Update_f_k(f_ks,ks,pos, tau, g, kmags = kcopy)

        # Change weight
        ke, pot, eloc = ke_pot_tot_energies(pos, wf, ham, newf_ks)
        weight *= np.exp(-tau * (eloc - eref))
        f_ks = newf_ks.copy()
        
        # Branch
        wtot = np.sum(weight)
        wavg = wtot / nconfig

        #branching lets us split the walkers with too-large weights and kill the walkers with too-small weights; want to keep # walkers = const
        probs = np.cumsum(weight/wtot) #stack up weights to find cumulative probabilities over each step - sums to 1
        randnums = np.random.random(nconfig) #throw random numbers b/w 0 and 1 for each walker
        new_idxs = np.searchsorted(probs, randnums) #find indices where the random number walkers would fit into the probs array
        posnew = pos[:,:,new_idxs] #update walkers, AKA slicing; selects only those walkers with weights between 0 and 1, kills the rest. See Kristoffer-Oleson-DMC.pdf pg 14 for more details on branching
        pos = posnew.copy()

        # Update the reference energy
        eref = eref - np.log(wavg)

        if istep % 10 == 0:
            print(
                "iteration",
                istep,
                "average energy",
                np.mean(eloc * weight / wavg),
                "eref",
                eref,
                "acceptance",
                acc_ratio,
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
    #from slaterwf import ExponentSlaterWF
    from wavefunction import MultiplyWF, JastrowWF
    from hamiltonian import Hamiltonian

    nconfig = 100 #default is 5000
    dfs = []
    N = 10 #num of momenta
    L = 1 #sys size - units??
    g = 2*np.sqrt(np.pi*alpha*l/L**3)
    for tau in [0.01]: #,0.005, 0.0025]:
        dfs.append(
            simple_dmc(
                JastrowWF(0.5), 
                #MultiplyWF(ExponentSlaterWF(2.0), JastrowWF(0.5)),
                Hamiltonian(U=U_STO,g=g),
                pos=np.random.randn(2, 3, nconfig), 
                g=g, N=N, L=L,
                tau=tau,
                nstep=100, #orig: 10000
            )
        )
    df = pd.concat(dfs)
    df.to_csv("dmc.csv", index=False)
