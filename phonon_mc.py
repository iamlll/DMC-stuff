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
conv = 1E-9/1.602E-19 #convert statC^2 (expressions with elec^2) to eV
convJ = 1/1.602E-19 #convert J to eV
eta_STO = epsinf/epssr
alpha = (elec**2*1E-9)/hbar*np.sqrt(m/(2*hbar*w))*1/epsinf*(1 - epsinf/epssr) #convert statC to J*m
U_STO = elec**2/(epsinf*hbar)*np.sqrt(2*m/(hbar*w))*1E-9 #convert statC in e^2 term into J to make U dimensionless
Ry = m*elec**4*(1E-9)**2/(2*epsinf**2 *hbar**2)*1/1.602E-19 #Rydberg energy unit in media, eV
a0 = hbar**2*epsinf/(m*elec**2 *1E-9); #Bohr radius in media
l = np.sqrt(hbar/(2*m*w))/ a0 #phonon length in units of the Bohr radius  

#####################################

def eph_energies(pos, wf,ham, tau, h_ks,f_ks, ks, kcopy):
    """ calculate kinetic + Coulomb + electron-phonon and phonon energies
    Input:
      pos: electron positions (nelec,ndim,nconf) 
      wf: wavefunction
      ham: hamiltonian
      tau: timestep
      ks: allowed momentum values
      kcopy: array of k-vector magnitudes, (nconfig) x (# ks) matrix
    Return:
      ke: kinetic energy
      pot: Coulomb energy - a constant for fixed electrons
      ph: Phonon + electron-phonon (local) energies
    """
    #ke = - np.sum(wf.laplacian(pos), axis=0)
    pot = ham.pot_ee(pos)
    
    #find elec density matrix
    dprod1 = np.matmul(ks,pos[0,:,:]) #np array for each k value; k dot r1
    dprod2 = np.matmul(ks,pos[1,:,:]) #k dot r2 
    rho = np.exp(1j*dprod1) + np.exp(1j*dprod2) #electron density eikr1 + eikr2
    
    #Update f_k from H_ph and H_eph
    fp = f_ks* np.exp(-tau/l**2)
    f2p = fp - tau*1j* ham.g/kcopy * np.conj(rho) #f'' = f' - it*g/k* (rho*)

    #Update weights from H_ph and H_eph, and calculate local energy
    ph = -1./tau* (np.sum(tau*1j* ham.g * fp/kcopy*rho,axis=0) + np.sum( np.conj(h_ks)*(f2p - f_ks),axis=0) ) #sum over all k-values; weights are normalized
    return pot, ph, pot+ph, rho, f2p

def mixed_estimator(pos, rho, ham, h_ks, f_ks, kmag):
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
    H_eph = 1j* ham.g*np.sum( (-f_ks * rho + np.conj(h_ks) *np.conj(rho))/kmag , axis=0) #sum over all k values; f/kmag = (# ks) x nconfigs matrix
    #find H_ph
    fhmag = f_ks* np.conj(h_ks) #find f_k magnitudes
    H_ph = 1/l**2 * np.sum(fhmag,axis=0)
    return ham.pot_ee(pos) + H_eph + H_ph
    #return ham.pot_ee(pos)

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
def simple_dmc(wf, ham, tau, pos, nstep=1000, N=5, L=10):
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
    f_ks = init_f_k(ks, kmag, ham.g, nconfig)
    h_ks = f_ks #this describes our trial wave fxn coherent state amplitudes

    #eref = -0.17/Ry #initialize reference energy with our best guess for the Gaussian bipolaron binding energy (units of Ry)
    #eref = ham.pot_ee(pos) - 0.1
    coul, ph, E_0, rho, f2p = eph_energies(pos, wf, ham, tau, h_ks, f_ks, ks, kcopy)
    eref = np.mean(E_0)
    print(eref)

    for istep in range(nstep):
        # Drift+diffusion - no diffusion here since electrons are fixed in place
        #Update weights from H_ph and H_eph (and coulomb = const), and calculate local energy
        coul, ph, E_0, rho, f2p = eph_energies(pos, wf, ham, tau, h_ks, f_ks, ks, kcopy)
        E_mix = mixed_estimator(pos, rho, ham, h_ks, f_ks, kcopy) #mixed estimator formulation of energy
        f_ks = f2p
        
        # Branch 
        weight = weight* np.exp(-tau * (E_0 - eref))
        wtot = np.sum(weight)
        wavg = wtot / nconfig
        E_gth = -1.*np.log(wavg)
        
        # Update the reference energy
        eref = eref + E_gth #this is literally just setting eref = E_0 in a very roundabout way
        #eref = E_0

        if istep % 1 == 0:
            print(
                "iteration",
                istep,
                "average weight",
                wavg.real,
                "E_mix",
                np.mean(E_mix).real,
                "E_0", #at this particular step, per config; should converge to E_mix as tau -> 0. In this toy problem E_0 = E_loc = E_ref
                #np.mean(eloc * weight / wavg),
                np.mean(E_0).real,
                "eref",
                eref.real,
                "E_gth", #want E_gth = 0 as tau -> inf
                E_gth.real,
                #"f_k0",
                #f_ks[0][0],
            )
        
        df["step"].append(istep)
        df["elocal"].append(np.mean(E_0))
        df["weight"].append(np.mean(weight))
        df["elocalvar"].append(np.std(E_0))
        df["weightvar"].append(np.std(weight))
        df["eref"].append(eref)
        df["tau"].append(tau)
        
        #Branching lets us split the walkers with too-large weights and kill the walkers with too-small weights; want to keep # walkers = const
        #The growth estimator should be done between population control calls. That is, evaluate a w' and w, compute their ratio, without calling "comb". Only call "comb" to do population control once every few steps (say every 10 steps)!!
        if istep % 50 == 0:
            
            #DMC tutorial comb algorithm
            probs = np.cumsum(weight/wtot) #stack up weights from each walker to find cumulative probabilities over each step - sums to 1 (i.e. last elt = 1)
            randnums = np.random.random(nconfig) #throw random numbers b/w 0 and 1 for each walker
            new_idxs = np.searchsorted(probs, randnums) #find indices where the random number walkers would fit into the probs array
            pos = pos[:,:,new_idxs] #update walkers, AKA slicing; selects only those walkers with weights between 0 and 1, kills the rest
            weight.fill(wavg)
            
            '''
            #RMP_DMC article suggestion for branching algorithm
            newwalkers = weight + np.random.uniform(high=1.,size=nconfig) #number of walkers progressing to next step at each position
            newwalkers = np.array(list(map(int,newwalkers)))

            #find indices where number of new walkers > 0; configs where newwalkers = 0 get killed off
            new_idxs = np.where(newwalkers >0)[0]
            newpos = pos[:,:,new_idxs] #newpos contains only configs which contain >= 1 walker. Now want to append positions to this
            newwts = weight[new_idxs]
            newfs = f_ks[:,new_idxs]
            #now append new walkers to our basis state arrays (position, f_k's, weights)
            for i in np.where(newwalkers >1)[0]:
                for num in range(newwalkers[i]-1):
                    newpos = np.append(newpos, pos[:,:,i][:,:,np.newaxis], axis=2)
                    #need to update weights as well
                    newwts = np.append(newwts, np.array([weight[i]]))
                    newfs = np.append(newfs, f_ks[:,i][:,np.newaxis], axis=1)
            pos = newpos
            weight = newwts
            f_ks = newfs
            nconfig = pos.shape[-1]
            kcopy = np.array([[ kmag[i] for j in range(nconfig)] for i in range(len(kmag))])
            h_ks = init_f_k(ks, kmag, g,nconfig) 
            '''
        #weight.fill(wavg) #use if no branching

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
    U = 2.

    for tau in [0.01]: #,0.005, 0.0025]:
        dfs.append(
            simple_dmc(
                JastrowWF(0.5), 
                #MultiplyWF(ExponentSlaterWF(2.0), JastrowWF(0.5)),
                Hamiltonian(g=g, hw=1/l**2),
                pos=np.random.randn(2, 3, nconfig), 
                N=N, L=L,
                tau=tau,
                nstep=3000, #orig: 10000
            )
        )
    df = pd.concat(dfs)
    df.to_csv("phonon_mc.csv", index=False)
