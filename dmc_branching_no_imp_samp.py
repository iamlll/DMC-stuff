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

def elec_energies(pos,wf,ham):
    """ calculate kinetic + Coulomb energies
    Input:
      pos: electron positions (nelec,ndim,nconf) 
      wf: wavefunction
      ham: hamiltonian
    Return:
      ke: kinetic energy
      pot: Coulomb energy - a constant for fixed electrons
      eloc: local energy
    """
    ke = - np.sum(wf.laplacian(pos), axis=0)
    pot = ham.pot_ee(pos)
    return ke+pot

def mixed_estimator(ham, g, pos, rho, kmag, h_ks, f_ks):
    '''
    Calculate energy using the mixed estimator form E_0 = <psi_T| H |phi>, psi_T & phi are coherent states
    Input:
        
        pos: electron positions (nelec, ndim, nconfigs)
        rho: electron density (eikr1 + eikr2)
        kmag: k-vector magnitudes, matrix size (len(ks), nconfigs)
        h_ks: coherent state amplitudes of trial wave function psi_T (len(ks), nconfigs)
        f_ks: coherent state amplitudes of our time-evolved numerical coherent state |{f_k}>
    '''
    #Find electron phonon energy
    H_eph = 1j*g*np.sum( (-f_ks * rho + np.conj(h_ks) *np.conj(rho))/kmag , axis=0) #sum over all k values; f/kmag = (# ks) x nconfigs matrix
    #find H_ph
    fhmag = f_ks* np.conj(h_ks) #find f_k magnitudes
    H_ph = 1/l**2 * np.sum(fhmag,axis=0)
    return ham.pot_ee(pos) + H_eph + H_ph

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
    #print((ratio[0],gfratio[0]))
    return np.minimum(1,ratio * gfratio)

def init_f_k(ks, kmag, g, nconfig):
    '''
    Initialize the phonon displacement functions f_k from the optimized Gaussian result
    input:
        ks: allowed k-vectors in the supercell
    '''
    #find f_ks
    yopt = 1.39
    sopt = 1.05E-9/a0 #in units of the Bohr radius
    d = yopt*sopt #assume pointing in z direction
    f_ks = -2j*g*l**2/kmag* np.exp(-kmag**2 * sopt**2/4) * (np.cos(ks[:,2] * d/2) - np.exp(-yopt**2/2) )/(1- np.exp(-yopt**2/2))
    f_kcopy = np.array([[ f_ks[i] for j in range(nconfig)] for i in range(len(ks))]) #make f_ks array size (# ks) x (# configurations)
    return f_kcopy

from itertools import product
def simple_dmc(wf, ham, tau, pos, g,nstep=1000, N=5, L=10):
    """
  Inputs:
  g: DOS for el-ph interaction
  w: LO phonon freq
  N: number of allowed k-vals in each direction
  L: box length (units of a0)
 
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
    #pos, acc = metropolis_sample(pos, wf, tau=0.5)
    weight = np.ones(nconfig)
    #Make a supercell/box
    #k = (nx, ny, nz)*2*pi/L for nx^2+ny^2+nz^2 <= n_c^2 for cutoff value n_c = N, where n_c -> inf is the continuum limit. 
    #A k-sphere cutoff is conventional as it specifies a unique KE cutoff
    ks = 2*np.pi/L* np.array([[nx,ny,nz] for nx,ny,nz in product(range(1,N+1), range(1,N+1), range(1,N+1)) if nx**2+ny**2+nz**2 <= N**2 ])

    kmag = np.sum(ks**2,axis=1)**0.5 #find k magnitudes
    kcopy = np.array([[ kmag[i] for j in range(nconfig)] for i in range(len(kmag))])
    #initialize f_ks
    f_ks = init_f_k(ks, kmag, g,nconfig)
    h_ks = f_ks #this describes our trial wave fxn coherent state amplitudes

    eref = -0.17/Ry #initialize reference energy with our best guess for the Gaussian bipolaron binding energy (units of Ry)

    for istep in range(nstep):
        # Drift+diffusion - no diffusion here since electrons are fixed in place
        pos = pos + np.sqrt(tau) * np.random.randn(*pos.shape) #update positions

        #update weights from KE + H_coul
        weight *= np.exp(-tau * elec_energies(pos, wf, ham))
        
        dprod1 = np.matmul(ks,pos[0,:,:]) #np array for each k value; k dot r1
        dprod2 = np.matmul(ks,pos[1,:,:]) #k dot r2 
        rho = np.exp(1j*dprod1) + np.exp(1j*dprod2) #electron density eikr1 + eikr2

        #Update f_k from H_ph and H_eph
        fp = f_ks* np.exp(-tau/l**2)
        f2p = fp - tau*1j*g/kcopy * np.conj(rho) #f'' = f' - it*g/k* (rho*)

        #Update weights from H_ph and H_eph, and calculate local energy
        weight = weight* np.exp(np.sum(tau*1j*g * fp/kcopy*rho,axis=0)) #from exp(-t*H_{eph}) |f>
        weight = weight* np.exp(np.sum(np.conj(h_ks)*(f2p - f_ks),axis=0)) #from exp(-t*H_{ph}) |f> and normalization <h|f> ~ exp(\sum h* f)
        eloc = -1/tau*np.log(weight) #growth estimator formulation of energy: E_0 = E_gth + E_ref = GS energy, target is E_ref = E_0 i.e. E_gth = 0
        weight *= np.exp(tau*eref)
        E_mix = mixed_estimator(ham, g, pos, rho, kcopy, h_ks, f_ks) #mixed estimator formulation of energy
        f_ks = f2p
        
        # Branch
        wtot = np.sum(weight)
        wavg = wtot / nconfig
        E_gth = -1./tau*np.log(wavg)
        
        #branching lets us split the walkers with too-large weights and kill the walkers with too-small weights; want to keep # walkers = const
        probs = np.cumsum(weight/wtot) #stack up weights to find cumulative probabilities over each step - sums to 1
        randnums = np.random.random(nconfig) #throw random numbers b/w 0 and 1 for each walker
        new_idxs = np.searchsorted(probs, randnums) #find indices where the random number walkers would fit into the probs array
        pos = pos[:,:,new_idxs] #update walkers, AKA slicing; selects only those walkers with weights between 0 and 1, kills the rest. See Kristoffer-Oleson-DMC.pdf pg 14 for more details on branching
         
        # Update the reference energy
        eref = eref + E_gth #this is literally just setting eref = E_0 in a very roundabout way
        
        if istep % 100 == 0:
            print(
                "iteration",
                istep,
                "average weight",
                wavg,
                "E_mix",
                np.mean(E_mix),
                "E_0", #at this particular step, per config; should converge to E_mix as tau -> 0
                #np.mean(eloc * weight / wavg),
                np.mean(eloc),
                "E_gth", #want E_gth = 0 as tau -> inf
                E_gth,
                "eref",
                eref,
                #"f_k0",
                #f_ks[0][0],
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
        #weight.fill(1.)

    return pd.DataFrame(df)


#####################################

if __name__ == "__main__":
    #from slaterwf import ExponentSlaterWF
    from wavefunction import MultiplyWF, JastrowWF
    from hamiltonian import Hamiltonian

    nconfig = 10 #default is 5000, we only need one since there's no randomness/branching going on yet
    dfs = []
    N = 10 #num of momenta
    L = 5 #sys size/length measured in a0
    g = 2/l**2 *np.sqrt(np.pi*alpha* l/L**3)
    U = 2/epsinf

    for tau in [0.01]: #,0.005, 0.0025]:
        dfs.append(
            simple_dmc(
                JastrowWF(0.5), 
                #MultiplyWF(ExponentSlaterWF(2.0), JastrowWF(0.5)),
                Hamiltonian(U= U,g=g,hw=1/l**2),
                pos=np.random.randn(2, 3, nconfig), 
                g=g,N=N, L=L,
                tau=tau,
                nstep=10000, #orig: 10000
            )
        )
    df = pd.concat(dfs)
    df.to_csv("dmc_branching_no_imp_samp.csv", index=False)
