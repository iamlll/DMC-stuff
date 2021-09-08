'''
Testing Ewald + jellium routines considering only KE + Coulomb energies (no phonons)
'''

#!/usr/bin/env python
import numpy as np
import sys

from metropolis import metropolis_sample
import pandas as pd
import matplotlib.pyplot as plt

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

def PBCjell_E(pos, wf, ham):
    '''returns kinetic energy, Ewald energy (e-e only), and total potential in Rydbergs using the distance between the two electrons as input (i.e. applying minimum image convention PBC)'''

    ke = -np.sum(wf.laplacian(pos), axis=0)
    return ke, 0., ke
    #return ke, ham.ewald(pos), ke + ham.ewald(pos)

def Test_Jastrow(wf, ham, nconfig=5):
    initpos = np.zeros((2,3,nconfig))
    xs = (wf.L)*np.random.rand(nconfig)
    #xs = np.where(xs >= wf.L, xs - wf.L, xs)
    #xs = np.where(xs < 0, xs + wf.L, xs)

    initpos[1,0,:] = xs

    bins = np.linspace(0,L,5)
    Xs = np.ravel(initpos[:,0,:])
    Ys = np.ravel(initpos[:,1,:])
    hist,xbins,ybins = np.histogram2d(Xs,Ys,bins=bins)
    fig2 = plt.figure()
    axhist = fig2.add_subplot(111, title='pcolormesh: actual edges', aspect='equal')
    Xbins, Ybins = np.meshgrid(xbins, ybins)
    cp = axhist.pcolormesh(Xbins, Ybins, hist)
    cbar=fig2.colorbar(cp) # Add a colorbar to a plot
    cbar.ax.set_ylabel("number")

    v = wf.value(initpos)
    g = wf.gradient(initpos)[0,0,:] #just take the positive x deriv
    l = wf.nabla2(initpos)[0]
    _,_,eloc = PBCjell_E(initpos,wf,ham)
    #Plot v,g,l,E vs x. v,g,l should satisfy PBC; eloc should not diverge as x->0 
    #fig = plt.figure(figsize=(6,4.5))
    #ax = fig.add_subplot(111)
    fig, ax = plt.subplots(2, 2, sharex=True) 
    ax[0,0].plot(xs,v,'b.',label='value')
    ax[0,1].plot(xs,g,'r.',label='gradient')
    ax[1,0].plot(xs,l,'g.',label='laplacian')
    ax[1,1].plot(xs,eloc,'k.',label='eloc')
    ax[0,0].legend()
    ax[0,0].set_xlabel('x')
    ax[0,1].legend()
    ax[0,1].set_xlabel('x')
    ax[1,0].legend()
    ax[1,0].set_xlabel('x')
    ax[1,1].legend()
    ax[1,1].set_xlabel('x')
    fig.subplots_adjust(hspace=0.025)
    plt.show()

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

def hist_reblock(pos, bins):
    '''
    Accumulate 3D histogram of electron positions; each bin encompasses some range of positions, and each entry (+1) corresponds to a (any) walker being within that bin. This is NOT the quantum mechanical probability of the electron but rather the stochastic prob of being in a particular location
    See https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
    '''
    #collect all x and y values in pos array 
    xs = np.ravel(pos[:,0,:])
    ys = np.ravel(pos[:,1,:])
    #now make x and y coords each a 1D array 
    hist,xbin,ybin_ = np.histogram2d(xs,ys,bins=bins) #also returns xbins and ybins, which aren't helpful since I already know what the binning is
    # Histogram does not follow Cartesian convention (see Notes),
    # therefore transpose H for visualization purposes.
    return hist.T       

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
        "ke": [],
        "pot": [],
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

    _,_,eloc = PBCjell_E(pos, wf, ham)
    eref = np.mean(eloc)
    print(eref)

    for istep in range(nstep):
        #rdist = np.mean(np.sum((pos[0,:,:]-pos[1,:,:])**2,axis=0)**0.5) #mean distance between the two electrons
        
        driftold = tau * wf.gradient(pos)
        _,_,elocold = PBCjell_E(pos, wf, ham)

        # Drift+diffusion 
        #with importance sampling
        posnew = pos + np.sqrt(tau) * np.random.randn(*pos.shape) + driftold
        driftnew = tau * wf.gradient(posnew)
        acc = acceptance(pos, posnew, driftold, driftnew, tau, wf)
        imove = acc > np.random.random(nconfig)
        pos[:, :, imove] = posnew[:, :, imove]
        acc_ratio = np.sum(imove) / nconfig

        ke,ewald,eloc = PBCjell_E(pos, wf, ham)
        
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
                "ke", np.mean(ke), "ewald", np.mean(ewald),
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
        df["pot"].append(np.mean(ewald))
        df["ke"].append(np.mean(ke))
        df["elocal"].append(np.mean(eloc))
        df["weight"].append(np.mean(weight))
        df["elocalvar"].append(np.std(eloc))
        df["weightvar"].append(np.std(weight))
        df["eref"].append(eref)
        df["tau"].append(tau)
        df["r_s"].append(r_s)
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
    nconfig = pos.shape[2]
    weight = np.ones(nconfig)

    _,_,eloc = PBCjell_E(pos, wf, ham)
    eref = np.mean(eloc)
    print(eref)

    blocksize = 1 #units of Bohr radius a0
    nblocks = int(L/blocksize)
    bins = np.linspace(0,L,nblocks)
    print(bins)
    hist = hist_reblock(pos, bins)
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
        hist = hist + hist_reblock(pos, bins)

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

    fig2 = plt.figure()
    axhist = fig2.add_subplot(111, title='e- density', aspect='equal')
    xbins, ybins = np.meshgrid(bins, bins)
    cp = axhist.pcolormesh(xbins, ybins, hist/sum(hist))
    cbar=fig2.colorbar(cp) # add a colorbar to a plot
    cbar.ax.set_ylabel("number")
    plt.show()
    return pd.DataFrame(df)

#####################################

if __name__ == "__main__":
    from slaterwf import ExponentSlaterWF
    from wavefunction import MultiplyWF, JastrowWF, UniformWF, PBCJastrowWF
    from ham import Hamiltonian
    import time

    tproj = 128 #projection time = tau * nsteps

    nconfig = 50000 #default is 5000
    dfs = []
    r_s = int(sys.argv[1]) #inter-electron spacing, controls density
    L = (4*np.pi*2/3)**(1/3) * r_s #sys size/length measured in a0; multiply by 2 since 2 = # of electrons
    print("L",L)
    U = 2.
    if len(sys.argv) > 2:
        alf = float(sys.argv[2])
    else: alf = 0.5
    csvname = "DMC_free_rs_" + str(r_s) + "_popsize_" + str(nconfig) + "_alpha_" + str(alf) +  ".csv"
    wf = PBCJastrowWF(alf,L, True)
    ham = Hamiltonian(U=U, L=L)
    #Test_Jastrow(wf, ham)

    np.random.seed(0)
    tic = time.perf_counter()
    print("VMC_jellium_rs_" + str(r_s) + ".csv")
   
    for tau in [r_s/20]: #[r_s/10, r_s/20, r_s/40, r_s/80]:
        #nstep = int(tproj/tau)
        nstep = 10000
        print(nstep)
        
        dfs.append(
            simple_dmc(
                wf,
                ham,
                pos= L* np.random.rand(2, 3, nconfig), 
                L=L,
                tau=tau,
                popstep=10,
                nstep=nstep #orig: 10000
            )
        )
        ''' 
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
       ''' 
    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")

    df = pd.concat(dfs)
    #df.to_csv("VMC_jellium_rs_" + str(r_s) + "_popsize_" + str(nconfig) + "_tproj_" + str(tproj) +  ".csv", index=False)
    df.to_csv(csvname, index=False)
    
