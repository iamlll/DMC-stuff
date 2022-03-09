'''
Testing electron + phonon DMC driver
'''

#!/usr/bin/env python
import numpy as np
import sys
sys.path.append("../")
from metropolis import metropolis_sample
import pandas as pd
import matplotlib.pyplot as plt
from qharv.reel import config_h5
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

def gth_estimator(ke_coul, pos, wf,configs,g, tau, h_ks,f_ks, ks, kcopy,phonon=True):
    """ calculate kinetic + Coulomb + electron-phonon and phonon energies in growth estimator formulation
    Input:
      ke_coul: kinetic+Coulomb energy for its shape
      pos: electron positions (nconf,nelec,ndim) 
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
    if phonon == True:
        #swap 1st and 3rd axes in pos matrix so ks dot r1 = (Nx3) dot (3 x nconf) = N x nconf matrix 
        swappos = np.swapaxes(pos,0,2)

        #find elec density matrix
        dprod1 = np.matmul(ks,swappos[:,0,:]) #np array for each k value; k dot r1
        dprod2 = np.matmul(ks,swappos[:,1,:]) #k dot r2 
        rho = np.exp(1j*dprod1) + np.exp(1j*dprod2) #electron density eikr1 + eikr2
        #Update f_k from H_ph and H_eph; [tau] = 1/ha
        fp = f_ks* np.exp(-tau/(2*l**2))
        f2p = fp - 1j*tau* g/kcopy * np.conj(rho) #f'' = f' - it*g/k* (rho*)
    
        #Update weights from H_ph and H_eph, and calculate local energy
        ph = -1./tau* (np.sum( tau*1j* g * fp/kcopy*rho,axis=0) + np.sum( np.conj(h_ks)*(f2p-f_ks),axis=0) ) #sum over all k-values; coherent state weight contributions are normalized
    else:
        f2p = np.zeros(f_ks.shape)
        ph = np.zeros(ke_coul.shape)
    return ke_coul+ph, f2p

def update_f_ks(pos, wf,g, tau, h_ks,f_ks, ks, kcopy,phonon=True):
    """ calculate electron density and update phonon coherence amplitudes.
    Input:
      pos: electron positions (nconf,nelec,ndim) 
      wf: wavefunction
      g: density of states of electron-phonon interaction
      tau: timestep
      ks: allowed momentum values
      kcopy: array of k-vector magnitudes, (nconfig) x (# ks) matrix
    Return:
      rho: electron density
      newf_ks: updated coherence state amplitudes
    """
    if phonon == True:
        #swap 1st and 3rd axes in pos matrix so ks dot r1 = (Nx3) dot (3 x nconf) = N x nconf matrix 
        swappos = np.swapaxes(pos,0,2)

        #find elec density matrix
        dprod1 = np.matmul(ks,swappos[:,0,:]) #np array for each k value; k dot r1
        dprod2 = np.matmul(ks,swappos[:,1,:]) #k dot r2 
        rho = np.exp(1j*dprod1) + np.exp(1j*dprod2) #electron density eikr1 + eikr2
        #Update f_k from H_ph and H_eph; [tau] = 1/ha
        newf_ks = f_ks* np.exp(-tau/(2*l**2)) - 1j*tau* g/kcopy * np.conj(rho) #f'' = f' - it*g/k* (rho*)
    else:
        rho = np.zeros((len(ks),pos.shape[0])) 
        newf_ks = np.zeros(f_ks.shape)
    return rho, newf_ks

def mixed_estimator(ke_coul, pos, wf, configs, rho, g, h_ks, f_ks, kmag,phonon=True):
    '''
    Calculate energy (in ha) using the mixed estimator form E_0 = <psi_T| H |phi>, psi_T & phi are coherent states
    Also syncs DMC driver configs with internal wf electron configurations (GetEnergy)
    Input:
        ke_coul: kinetic+Coulomb energy for its shape
        pos: electron positions (nelec, ndim, nconfigs)
        rho: electron density (eikr1 + eikr2)
        kmag: k-vector magnitudes, matrix size (len(ks), nconfigs)
        h_ks: coherent state amplitudes of trial wave function psi_T (len(ks), nconfigs)
        f_ks: coherent state amplitudes of our time-evolved numerical coherent state |{f_k}>
    Output:
        total energy
    '''
    #Find electron phonon energy
    if phonon == True:
        H_eph = 1j* g*np.sum( (-f_ks * rho + np.conj(h_ks) *np.conj(rho))/kmag , axis=0) #sum over all k values; f/kmag = (# ks) x nconfigs matrix. See eqn 
        #find H_ph
        H_ph = 1/(2*l**2) * np.sum(f_ks* np.conj(h_ks),axis=0)
    else:
        H_eph = np.zeros(ke_coul.shape)
        H_ph = np.zeros(ke_coul.shape)
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
    f_ks = -4j*g*l**2/kmag* np.exp(-kmag**2 * sopt**2/4) * (np.cos(ks[:,2] * d/2) - np.exp(-yopt**2/2) )/(1- np.exp(-yopt**2/2))
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

def plotamps(kcopy, n_ks, N):
    # f_ks: (# ks) x (nconfig) array of coherent state amplitudes. Want to make histogram of f_ks vs |k| for final config of f_ks.
    fig,ax = plt.subplots(2,1)
    ax[0].plot(kcopy.flatten(), n_ks.real.flatten(),'.')
    ax[0].set_xlabel('$|\\vec{k}|$')
    ax[0].set_ylabel('Re($n_k$)')
    ax[1].plot(kcopy.flatten(), n_ks.imag.flatten(),'.')
    ax[1].set_xlabel('$|\\vec{k}|$')
    ax[1].set_ylabel('Im($n_k$)')
    ax[0].set_ylim(bottom=0)
    ax[1].set_ylim(bottom=0)
    fig.suptitle('$N = $' + str(N))

    plt.tight_layout()
    plt.show()
    #want to find the relationship between k_cut and L 

def InitPos(wf,opt='rand'):
    if opt == 'bcc':
        #initialize one electron at center of box and the other at the corner
        pos= np.zeros((wf.nconfig, wf.nelec, wf.ndim))
        pos0 = np.full((wf.nconfig, wf.ndim),wf.L/2)
        pos1 = np.full((wf.nconfig, wf.ndim),wf.L)
        pos[:,0,:] = pos0
        pos[:,1,:] = pos1
    else:    
        pos= wf.L* np.random.rand(wf.nconfig, wf.nelec, wf.ndim)
    return pos

from itertools import product
def simple_dmc(wf, tau, pos, popstep=1, nstep=1000, N=5, L=10,elec=True,phonon=True,l=l,eta=eta_STO,gth=True,h5name="dmc.h5"):
    """
  Inputs:
  L: box length (units of a0)
  pos: initial position
  nstep: total number of steps in the sim
  N: number of allowed k-vals in each direction
  Outputs:
  A Pandas dataframe with each 
  """
    from time import time
    df = {
        "step": [],
        "nconfig": [],
        "r_s": [],
        "N_cut": [], #spherical momentum cutoff radius determining max k-vector magnitude
        "l":[], #phonon length scale
        "L":[], #system size
        "eta":[],
        "g": [],
        "ke_coul": [],
        "elocal": [],
        "egth": [],
        "weight": [],
        "weightvar": [],
        "elocalvar": [],
        "eref": [],
        "tau": [],
        "popstep": [],
        #"f_ks": [], #avg phonon amplitudes
        #"h_ks":[], #initial trial wf guess for phonon amps
        #"n_ks": [], #avg momentum density hw a*a
        #"ks": [],
    }
    # use HDF file for large data output
    h5file = config_h5.open_write(h5name)

    alpha = (1-eta)*l
    print('alpha',alpha)
    L = wf.L
    g = 1./l**2*np.sqrt(np.pi*alpha*l/L**3) #DOS, all lengths in units of Bohr radii a0

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
    if phonon == False: f_ks.fill(0.)
    h_ks = f_ks #this describes our trial wave fxn coherent state amplitudes
    #print(h_ks)
    #egth,_ = gth_estimator(pos, wf, configs, g, tau,h_ks, f_ks, ks, kcopy,phonon)
    #print(np.mean(eloc))
    rho, _ = update_f_ks(pos, wf, g, tau, h_ks, f_ks, ks, kcopy,phonon)
    ke_coul = GetEnergy(wf,configs,pos,'total')
    elocold = mixed_estimator(ke_coul, pos, wf, configs, rho, g, h_ks, f_ks, kcopy,phonon)

    eref = np.mean(elocold)
    print(eref)

    timers = dict(
      drift_diffusion = 0.0,
      mixed_estimator = 0.0,
      gth_estimator = 0.0,
      update_coherent = 0.0,
      branch = 0.0,
    )
    for istep in range(nstep):
        tick = time()
        if elec == True:
            driftold = tau * wf.grad(pos)

            # Drift+diffusion 
            #with importance sampling
            posnew = pos + np.sqrt(tau) * np.random.randn(*pos.shape) + driftold
            driftnew = tau * wf.grad(posnew)
            acc = acceptance(pos, posnew, driftold, driftnew, tau, wf)
            imove = acc > np.random.random(nconfig)
            pos[imove,:, :] = posnew[imove,:, :]
            acc_ratio = np.sum(imove) / nconfig
        tock = time()
        timers['drift_diffusion'] += tock - tick

        #update coherent state amplitudes
        tick = time()
        rho, f2p = update_f_ks(pos, wf, g, tau, h_ks, f_ks, ks, kcopy,phonon)
        tock = time()
        timers['update_coherent'] += tock - tick

        #compute observables
        tick = time()
        ke_coul = GetEnergy(wf,configs,pos,'total') #syncs internal wf configs object + driver configs object
        eloc = mixed_estimator(ke_coul, pos, wf, configs, rho, g, h_ks, f_ks, kcopy,phonon)
        tock = time()
        timers['mixed_estimator'] += tock - tick
        tick = time()
        if gth:
            egth,_ = gth_estimator(ke_coul, pos, wf, configs, g,tau, h_ks, f_ks, ks, kcopy,phonon)
        else: egth = np.zeros(eloc.shape)
        tock = time()
        timers['gth_estimator'] += tock - tick
        #syncs internal wf configs object + driver configs object
        f_ks = f2p
        n_ks = f_ks* np.conj(h_ks) #n_k = hw a*a; momentum distribution of equilibrium phonons -- want to plot this as a function of |k|

        oldwt = np.mean(weight)
        weight = weight* np.exp(-0.5* tau * (elocold + eloc - 2*eref))
        elocold = eloc
        
        # Branch
        tick = time()
        wtot = np.sum(weight)
        wavg = wtot / nconfig
        if elec == True:
            if istep % popstep == 0:
                pos, weight = popcontrol(pos, weight, wavg, wtot)
                wf.update(configs,pos)
        tock = time()
        timers['branch'] += tock - tick

        # Update the reference energy
        Delta = -1./tau* np.log(wavg/oldwt) #need to normalize <w_{n+1}>/<w_n>
        eref = eref + Delta

        if istep % popstep == 0:
            print(
                "iteration",
                istep,
                "avg wt",
                wavg.real,
                "ke_coul",
                np.mean(ke_coul),
                "average energy",
                np.mean(eloc * weight / wavg),
                "eref",
                eref,
                "sig_gth",
                np.std(eloc),
                "f_k avg",
                np.mean(f_ks[2,:]), #avg coh state amp for 3rd mom val
            )

        df['g'].append(g)
        df['l'].append(l)
        df['eta'].append(eta)
        df['N_cut'].append(N)
        df["step"].append(istep)
        df["ke_coul"].append(np.mean(ke_coul))
        df["elocal"].append(np.mean(eloc))
        df["egth"].append(np.mean(egth))
        df["weight"].append(np.mean(weight))
        df["elocalvar"].append(np.std(eloc))
        df["weightvar"].append(np.std(weight))
        df["eref"].append(eref)
        df["tau"].append(tau)
        df["r_s"].append(r_s)
        df["L"].append(L)
        df["nconfig"].append(nconfig)
        df['popstep'].append(popstep)
        #df['h_ks'].append(np.mean(h_ks,axis=1))
        #df['f_ks'].append(np.mean(f_ks,axis=1))
        #df['n_ks'].append(np.mean(n_ks,axis=1)) #avg over all walkers
        #df['ks'].append(kmag)
        grp = h5file.create_group(h5file.root, 's%08d' % istep)
        big_data = {
          'n_ks': n_ks.mean(axis=1),
          'f_ks': f_ks.mean(axis=1),
        }
        config_h5.save_dict(big_data, h5file, slab=grp)
    config_h5.save_dict({'ks': kmag, 'h_ks': h_ks.mean(axis=1)}, h5file)
    h5file.close()
    print('Timings:')
    for key, val in timers.items():
      line = '%16s %.4f' % (key, val)
      print(line)
    #plotamps(kcopy,n_ks, N)
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
        rho, f2p = update_f_ks(pos, wf, g, tau, h_ks, f_ks, ks, kcopy, phonons=False)
        ke = GetEnergy(wf,configs,pos,'ke') #syncs internal wf configs object + driver configs object
        eloc = mixed_estimator(pos, wf, configs, rho, g, h_ks, f_ks, kcopy,phonons=False)
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
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--rs', type=int, default=4)
    parser.add_argument('--nconf', type=int, default=512)
    parser.add_argument('--seed',type=int,default=0) #random seed
    parser.add_argument('--elec', type=int, default=1) #on/off switch for electrons
    parser.add_argument('--ph', type=int, default=1) #on/off switch for phonons
    parser.add_argument('--Ncut',type=int,default=10) # defines spherical momentum cutoff k_cut = 2pi*N/L
    parser.add_argument('--tproj',type=int,default=128) # projection time = tau * nsteps
    parser.add_argument('--l',type=int,default=l) 
    parser.add_argument('--eta',type=np.float64,default=eta_STO) 
    parser.add_argument('--gth',type=int,default=1) #on/off switch for growth estimator
    args = parser.parse_args()

    r_s = args.rs  # inter-electron spacing, controls density
    nconfig = args.nconf #default is 5000
    seed = args.seed
    elec_bool = args.elec > 0
    ph_bool = args.ph > 0
    gth_bool = args.gth > 0
    N = args.Ncut
    tproj = args.tproj #projection time = tau * nsteps

    dfs = []
    wf = UpdatedJastrow(r_s,nconfig=nconfig)
    print(wf.L)

    # Modify the Frohlich coupling constant alpha = (1-eta)*\tilde l
    l = args.l
    eta = args.eta
    #l = 20
    #eta = 0.2
    filename = "phonons_rs_{0}_popsize_{1}_seed_{2}_N_{3}_eta_{4:d}_U_{5:d}".format(r_s, nconfig, seed, N,int(eta),int(l))
    if elec_bool:
        filename = 'DMC_bcc_with_' + filename
    else:
        filename = 'DMC_no_elec_with_' + filename
    h5name = filename + ".h5"
    print(filename)
    print('elec',elec_bool)
    print('ph',ph_bool)
    print('gth',gth_bool)
   
    LLP = -alpha/(2*l**2) #-alpha hw energy lowering for single polaron
    feyn = (-alpha -0.98*(alpha/10)**2 -0.6*(alpha/10)**3)/(2*l**2)
    print('N',N)
    print('LLP',2*LLP)
    print('Feyn',feyn)
    np.random.seed(seed)
    tic = time.perf_counter()
    initpos = InitPos(wf,'bcc') 
    #for tau in [r_s/20, r_s/40, r_s/80]:
    for tau in [r_s/80]:
        nstep = int(tproj/tau)
        print(nstep)
        
        dfs.append(
            simple_dmc(
                wf,
                pos= initpos,
                tau=tau,
                popstep=10,
                N=N,
                nstep=nstep, #orig: 10000
                l=l,
                eta=eta,
                elec=elec_bool,
                phonon=ph_bool,
                gth=gth_bool,
                h5name = h5name,
            )
        )
    csvname = filename + ".csv"
    picklename = filename + ".pkl"
       
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
    #df.to_pickle(picklename)
