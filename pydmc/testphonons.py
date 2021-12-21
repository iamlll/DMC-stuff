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
'''phonon energy calculations'''

def update_f_ks(pos, wf,g, tau, h_ks,f_ks, ks, kcopy):
    """ calculate electron density and update phonon coherence amplitudes.
    Input:
      pos: electron positions (nelec,ndim,nconf) 
      wf: wavefunction
      g: density of states of electron-phonon interaction
      tau: timestep
      ks: allowed momentum values
      kcopy: array of k-vector magnitudes, (nconfig) x (# ks) matrix
    Return:
      rho: electron density
      newf_ks: updated coherence state amplitudes
    """
    #swap 1st and 3rd axes in pos matrix so ks dot r1 = (Nx3) dot (3 x nconf) = N x nconf matrix 
    swappos = np.swapaxes(pos,0,2) #CHECK that this does what I think it does

    #find elec density matrix
    dprod1 = np.matmul(ks,swappos[:,0,:]) #np array for each k value; k dot r1
    dprod2 = np.matmul(ks,swappos[:,1,:]) #k dot r2 
    rho = np.exp(1j*dprod1) + np.exp(1j*dprod2) #electron density eikr1 + eikr2
    
    #Update f_k from H_ph and H_eph; [tau] = 1/ha
    newf_ks = f_ks* np.exp(-tau/(2*l**2)) - 1j*tau* g/kcopy * np.conj(rho) #f'' = f' - it*g/k* (rho*)
    return rho, newf_ks

def mixed_estimator(pos, wf, configs, rho, g, h_ks, f_ks, kmag):
    '''
    Calculate energy (in ha) using the mixed estimator form E_0 = <psi_T| H |phi>, psi_T & phi are coherent states
    Also syncs DMC driver configs with internal wf electron configurations (GetEnergy)
    Input:
        pos: electron positions (nelec, ndim, nconfigs)
        rho: electron density (eikr1 + eikr2)
        kmag: k-vector magnitudes, matrix size (len(ks), nconfigs)
        h_ks: coherent state amplitudes of trial wave function psi_T (len(ks), nconfigs)
        f_ks: coherent state amplitudes of our time-evolved numerical coherent state |{f_k}>
    Output:
        total energy
    '''
    ke_coul = GetEnergy(wf,configs,pos,'total')
    #Find electron phonon energy
    H_eph = 1j* g*np.sum( (-f_ks * rho + np.conj(h_ks) *np.conj(rho))/kmag , axis=0) #sum over all k values; f/kmag = (# ks) x nconfigs matrix. See eqn 
    #find H_ph
    H_ph = 1/(2*l**2) * np.sum(f_ks* np.conj(h_ks),axis=0)
    return ke_coul + H_eph + H_ph

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
def simple_dmc(wf, g, tau, pos, popstep=1, nstep=1000, N=5, L=10):
    """
  Inputs:
  L: box length (units of a0)
  pos: initial position
  nstep: total number of steps in the sim
  N: number of allowed k-vals in each direction
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

    #Make a supercell/box
    #k = (nx, ny, nz)*2*pi/L for nx^2+ny^2+nz^2 <= n_c^2 for cutoff value n_c = N, where n_c -> inf is the continuum limit. 
    #A k-sphere cutoff is conventional as it specifies a unique KE cutoff
    ks = 2*np.pi/L* np.array([[nx,ny,nz] for nx,ny,nz in product(range(1,N+1), range(1,N+1), range(1,N+1)) if nx**2+ny**2+nz**2 <= N**2 ])

    kmag = np.sum(ks**2,axis=1)**0.5 #find k magnitudes
    kcopy = np.array([[ kmag[i] for j in range(nconfig)] for i in range(len(kmag))]) # (# ks) x nconfig matrix

    #initialize f_ks
    f_ks = init_f_k(ks, kmag, g, nconfig)
    h_ks = f_ks #this describes our trial wave fxn coherent state amplitudes

    rho, _ = update_f_ks(pos, wf, g, tau, h_ks, f_ks, ks, kcopy)
    eloc = mixed_estimator(pos, wf, configs, rho, g, h_ks, f_ks, kcopy)

    eref = np.mean(eloc)
    print(eref)

    for istep in range(nstep):
        driftold = tau * wf.grad(pos)
        elocold = mixed_estimator(pos, wf, configs, rho, g, h_ks, f_ks, kcopy)

        # Drift+diffusion 
        #with importance sampling
        posnew = pos + np.sqrt(tau) * np.random.randn(*pos.shape) + driftold
        driftnew = tau * wf.grad(posnew)
        acc = acceptance(pos, posnew, driftold, driftnew, tau, wf)
        imove = acc > np.random.random(nconfig)
        pos[imove,:, :] = posnew[imove,:, :]
        acc_ratio = np.sum(imove) / nconfig

        #update coherent state amplitudes
        rho, f2p = update_f_ks(pos, wf, g, tau, h_ks, f_ks, ks, kcopy)
        ke = GetEnergy(wf,configs,pos,'ke') #syncs internal wf configs object + driver configs object
        eloc = mixed_estimator(pos, wf, configs, rho, g, h_ks, f_ks, kcopy)
        #syncs internal wf configs object + driver configs object
        f_ks = f2p

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
                "f_k",
                f_ks[2,0], #coherent state amp for 1st walker of 3rd momentum value
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

def simple_vmc(wf, g, tau, pos, nstep=1000, N=10, L=10):
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
        "acceptance": [],
    }
    nconfig = pos.shape[0]
    weight = np.ones(nconfig)
    #setup wave function
    configs = wf.setup(pos)
    if nconfig != wf.nconfig:
        print("Incompatible number of walkers: sim nconfig = " + str(nconfig) + ", but wf nconfig = " + str(wf.nconfig) + ". Please re-run step1_opt.py for " + str(nconfig) + " walkers, then try again. Exiting program...")
        return

    #Make a supercell/box
    #k = (nx, ny, nz)*2*pi/L for nx^2+ny^2+nz^2 <= n_c^2 for cutoff value n_c = N, where n_c -> inf is the continuum limit. 
    #A k-sphere cutoff is conventional as it specifies a unique KE cutoff
    ks = 2*np.pi/L* np.array([[nx,ny,nz] for nx,ny,nz in product(range(1,N+1), range(1,N+1), range(1,N+1)) if nx**2+ny**2+nz**2 <= N**2 ])

    kmag = np.sum(ks**2,axis=1)**0.5 #find k magnitudes
    kcopy = np.array([[ kmag[i] for j in range(nconfig)] for i in range(len(kmag))]) # (# ks) x nconfig matrix

    #initialize f_ks
    f_ks = init_f_k(ks, kmag, g, nconfig)
    h_ks = f_ks #this describes our trial wave fxn coherent state amplitudes

    rho, _ = update_f_ks(pos, wf, g, tau, h_ks, f_ks, ks, kcopy)
    eloc = mixed_estimator(pos, wf, configs, rho, g, h_ks, f_ks, kcopy)

    eref = np.mean(eloc)
    print(eref)

    for istep in range(nstep):
        wfold=wf.val(pos)
        elocold = mixed_estimator(pos, wf, configs, rho, g, h_ks, f_ks, kcopy)
        # propose a move
        gauss_move_old = np.random.randn(*pos.shape)
        posnew=pos + np.sqrt(tau)*gauss_move_old

        wfnew=wf.val(posnew)
        # calculate Metropolis-Rosenbluth-Teller acceptance probability
        prob = wfnew**2/wfold**2 # for reversible moves
        # get indices of accepted moves
        acc_idx = (prob + np.random.random_sample(nconfig) > 1.0)
        # update stale stored values for accepted configurations
        pos[acc_idx,:,:] = posnew[acc_idx,:,:]
        wfold[acc_idx] = wfnew[acc_idx]
        acceptance = np.mean(acc_idx) #avg acceptance rate at each step (NOT total, would have to additionally divide by nstep)
        #update coherent state amplitudes
        rho, f2p = update_f_ks(pos, wf, g, tau, h_ks, f_ks, ks, kcopy)
        ke = GetEnergy(wf,configs,pos,'ke') #syncs internal wf configs object + driver configs object
        eloc = mixed_estimator(pos, wf, configs, rho, g, h_ks, f_ks, kcopy)
        #syncs internal wf configs object + driver configs object
        f_ks = f2p

        if istep % 10 == 0:
            print(
                "iteration",
                istep,
                "ke", np.mean(ke),
                "average energy",
                np.mean(eloc),
                "acceptance",acceptance
            )

        df["step"].append(istep)
        df["ke"].append(np.mean(ke))
        df["elocal"].append(np.mean(eloc))
        df["acceptance"].append(acceptance)
        df["tau"].append(tau)
        df["r_s"].append(r_s)

    return pd.DataFrame(df)

#####################################

if __name__ == "__main__":
    from updatedjastrow import UpdatedJastrow
    import time

    tproj = 128 #projection time = tau * nsteps

    nconfig = 512 #default is 5000
    dfs = []
    r_s = int(sys.argv[1]) #inter-electron spacing, controls density
    L = (4*np.pi*2/3)**(1/3) * r_s #sys size/length measured in a0; multiply by 2 since 2 = # of electrons
    print("L",L)
    N = 10 #number allowed momenta
    wf = UpdatedJastrow(r_s,nconfig=nconfig)
    g = 1./l**2*np.sqrt(np.pi*alpha*l/wf.L**3) #DOS, all lengths in units of Bohr radii a0
    seed = int(sys.argv[2])
    csvname = "phonons_rs_" + str(r_s) + "_popsize_" + str(nconfig) + "_seed_" + str(seed) + ".csv"

    np.random.seed(seed)
    tic = time.perf_counter()
     
    #for tau in [r_s/20, r_s/40, r_s/80]:
    for tau in [r_s/20]:
        nstep = int(tproj/tau)
        print(nstep)
        
        dfs.append(
            simple_dmc(
                wf,
                g,
                pos= L* np.random.rand(nconfig, 2, 3), 
                L=L,
                tau=tau,
                popstep=10,
                N=N,
                nstep=nstep #orig: 10000
            )
        )
    csvname = 'DMC_with_' + csvname
       
    ''' 
    for tau in [r_s/20]:
        nstep = int(tproj/tau)
        print(nstep)
        dfs.append(
            simple_vmc(
                wf,
                g,
                pos= L* np.random.rand(nconfig, 2, 3), 
                L=L,
                tau=tau,
                N=N,
                nstep=nstep #orig: 10000
            )
        )
    csvname = 'VMC_with_' + csvname
    '''    
    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")

    df = pd.concat(dfs)
    df.to_csv(csvname, index=False)
     
