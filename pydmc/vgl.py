import numpy as np
import sys
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from pyscf.pbc import gto as pbcgto
import pyqmc
from pyqmc.coord import PeriodicConfigs 
from egas import EnergyAccumulator
from pyqmc.wftools import generate_jastrow

def value(wf):
    expo = wf.value()[1]
    return np.exp(expo)

def gradient(configs,wf):
    '''
    configs: PeriodicConfigs object

    '''
    nconf, nelec, ndim = configs.configs.shape
    gradarr = np.zeros((nconf,nelec,ndim))
    for e in range(nelec):
        grad = np.swapaxes(wf.gradient(e, configs.electron(e)),0,1) #nconf x ndim array
        gradarr[:,e,:] = grad
    return gradarr

def laplacian(configs, wf):
    nconf, nelec, ndim = configs.configs.shape
    laparr = np.zeros((nelec,nconf))
    for e in range(nelec):
        lap = wf.gradient_laplacian(e, configs.electron(e))[1]
        laparr[e,:] = lap.real
    return laparr
    
def GenerateJastrow(filename,rs=4):
    '''create PyQMC JastrowSpin wf using optimized Jastrow parameters from step1_opt.py'''
    nelec = 2
    ndim = 3
    lbox = (4*np.pi/3*nelec)**(1/3) * rs  # sys. size/length measured in a0; multiply by 2 since 2 = # of electrons

    with h5py.File(filename, 'r') as f:
        bcoeff = f['wf']['bcoeff'][()]
        nconfig = f['nconfig'][0]

    axes = lbox*np.eye(ndim)
    initpos = np.zeros((nconfig,nelec,ndim))
    np.random.seed(0)
    # initialize electrons uniformly inside the box
    randL = lbox*np.random.rand(nconfig)
    #comput vgl, eloc along the x axis only 
    initpos[:,1,0] = randL

    # simulation cell
    cell = pbcgto.M(
      atom = 'He 0 0 0',
      a = axes,
      unit='B',  # B = units of Bohr radii
    )
    # ee Jastrow (only bcoeff for ee, no acoeff for ei)
    wf, to_opt = generate_jastrow(cell, ion_cusp=[], na=0)
    wf.parameters['bcoeff'] = bcoeff  # given by step1_opt.py

    configs = PeriodicConfigs(initpos, axes) #(nconf, nelec, ndim)
    
    wf.recompute(configs) #to obtain ._configscurrent needed to calc energy

    accumulator = {"energy": EnergyAccumulator(cell)} #energy accumulator, dict object storing energy info
    return wf, accumulator, configs

def plotVGL(filename,rs=4):
    '''compute value, gradient, laplacian for PyQMC JastrowSpin wf'''
    wf, accumulator,config = GenerateJastrow(filename,rs)
    val = value(wf)
    g = gradient(config,wf)[:,0,0] #pull out positive x derivative
    lap = laplacian(config, wf)[0]
    print(g.shape)
    '''
    setupstr="from vgl import GenerateJastrow, gradient, gradient2; filename='opt-rs4.h5'; rs=4; wf, accumulator,config = GenerateJastrow(filename,rs);"
    import timeit
    print(timeit.timeit(stmt="gradient(config,wf)",setup=setupstr,number=20))
    print(timeit.timeit(stmt="gradient2(config,wf)",setup=setupstr,number=20))
    '''

    # use hacked energy in estimator
    energy_acc = accumulator["energy"](config, wf)
    eloc = energy_acc["total"].real
    #print(eloc)
    xs = np.sqrt(np.sum((config.configs[:,0,:]-config.configs[:,1,:])**2,axis=1))
    print(xs.shape)
    #want to make wrapper/daughter class of JastrowSpin where I can save L, r_s into the wave function directly
    L = (4*np.pi/3*2)**(1/3) * rs  # sys. size/length measured in a0; multiply by 2 since 2 = # of electrons
    fig, ax = plt.subplots(2, 2, sharex=True) 
    ax[0,0].plot(xs,val,'b.',label='value')
    ax[0,1].plot(xs,g,'r.',label='gradient')
    ax[1,0].plot(xs,lap,'g.',label='laplacian')
    ax[1,1].plot(xs,eloc,'k.',label='eloc')
    ax[0,0].axvline(L/2)
    ax[0,1].axvline(L/2)
    ax[1,1].axvline(L/2)
    ax[1,0].axvline(L/2)
    ax[0,0].legend()
    ax[0,0].set_xlabel('r')
    ax[0,1].legend()
    ax[0,1].set_xlabel('r')
    ax[1,0].legend()
    ax[1,0].set_xlabel('r')
    ax[1,1].legend()
    ax[1,1].set_xlabel('r')
    fig.subplots_adjust(hspace=0.025)
    plt.show()
    
if __name__ == "__main__":
    plotVGL(sys.argv[1])
