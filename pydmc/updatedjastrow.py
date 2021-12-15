'''Convenience wrapper function / daughter class for the PyQMC JastrowSpin wave function'''

import numpy as np
import sys
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from pyscf.pbc import gto as pbcgto
import pyqmc
from pyqmc.coord import PeriodicConfigs 
from egas import EnergyAccumulator
from pyqmc.wftools import generate_jastrow, default_jastrow_basis
import pyqmc.jastrowspin as jastrowspin
import pyqmc.gpu as gpu
import copy

class UpdatedJastrow(jastrowspin.JastrowSpin):
    def __init__(self,rs):
        self.rs = rs
        self.nelec = 2
        self.ndim = 3
        self.L = (4*np.pi/3*self.nelec)**(1/3) * float(self.rs)  # sys. size/length measured in a0; multiply by 2 since 2 = # of electrons 

        axes = self.L*np.eye(self.ndim)
        #adapted from generate_jastrow
        # simulation cell
        cell = pbcgto.M(
          atom = 'He 0 0 0',
          a = axes,
          unit='B',  # B = units of Bohr radii
          )
        # ee Jastrow (only bcoeff for ee, no acoeff for ei)
        ion_cusp = []
        abasis, bbasis = default_jastrow_basis(cell, len(ion_cusp) > 0, na=0, nb=3, rcut=None)
        super().__init__(cell,a_basis=abasis, b_basis=bbasis)
        self.parameters["bcoeff"][0, [0, 1, 2]] = gpu.cp.array([-0.25, -0.50, -0.25])
        print(self._nelec)
        self.accumulator = {"energy": EnergyAccumulator(cell)} #energy accumulator, dict object storing energy info
    
    def setup(self, initpos):
        filename = 'opt-rs' + str(self.rs) + '.h5'
        with h5py.File(filename, 'r') as f:
            bcoeff = f['wf']['bcoeff'][()]
            self.nconfig = f['nconfig'][0]

        self.parameters['bcoeff'] = bcoeff  # given by step1_opt.py
        configs = PeriodicConfigs(initpos, self._mol.a) #(nconf, nelec, ndim)
        self.recompute(configs) #to obtain ._configscurrent needed to calc energy
        return configs

    def update(self,configs,pos=None):
        '''
        Syncs internal + DMC driver configs to position pos
        If pos not provided, syncs internal configurations to DMC driver ones
        '''
        if (pos is not None) & ((pos == configs.configs).all() == False):
            accept = [True] * self.nconfig
            newconfig = PeriodicConfigs(pos, self._mol.a) #(nconf, nelec, ndim)
            configs.move_all(newconfig, accept) #updates electronic configurations exposed to the user
        self.recompute(configs) #internally update configs to external configuration

    def val(self,pos):
        '''
        Calculates value of wf at pos - updates internal configs but NOT external ones! (to sync, run self.update() instead)
        updates internal configurations to calculate Jastrow wf value
        
        returns: Jastrow value evaluated at pos
        '''
        newconfig = PeriodicConfigs(pos, self._mol.a) #(nconf, nelec, ndim)
        expo = self.recompute(newconfig)[1] #internally updates both electrons to new position, and calculates Jastrow exponent; this is equivalent to running self.updateinternals twice (1x for each electron)
        return np.exp(expo)

    def grad(self,pos):
        '''
        configs: PeriodicConfigs object for electron positions (from driver)
        '''
        nconf, nelec, ndim = pos.shape
        gradarr = np.zeros((nconf,nelec,ndim))
        if (pos == self._configscurrent.configs).all() == False:
            #first update internal configs to be at pos if they're different
            newconfig = PeriodicConfigs(pos, self._mol.a) #(nconf, nelec, ndim)
            self.recompute(newconfig)
        for e in range(nelec):
            gradient = np.swapaxes(self.gradient(e, self._configscurrent.electron(e)),0,1) #nconf x ndim array; e denotes which electron is fixed (distance from other electron calculated from electron e)
            gradarr[:,e,:] = gradient
        return gradarr

    def lap(self, pos):
        nconf, nelec, ndim = pos.shape
        laparr = np.zeros((nelec,nconf))
        if (pos == self._configscurrent.configs).all() == False:
            #first update internal configs to be at pos if they're different
            newconfig = PeriodicConfigs(pos, self._mol.a) #(nconf, nelec, ndim)
            self.recompute(newconfig)
        for e in range(nelec):
            laplacian = self.gradient_laplacian(e, self._configscurrent.electron(e))[1]
            laparr[e,:] = laplacian.real
        return laparr
    
def GetEnergy(wf, config, pos, key='total'):
    #First update driver + internal configs to new position if necessary
    if ((config.configs == pos).all() == False) or ((wf._configscurrent.configs == pos).all() == False):
        wf.update(config, pos)
    energy_acc = wf.accumulator["energy"](config, wf)
    eloc = energy_acc[key].real
    return eloc

def plotVGL(wf):
    '''compute value, gradient, laplacian for PyQMC JastrowSpin wf'''
    #accumulator = {"energy": EnergyAccumulator(wf._mol)} #energy accumulator, dict object storing energy info
    nconfig = 512
    initpos = np.zeros((nconfig,wf._nelec,wf.ndim))
    np.random.seed(0)
    # initialize electrons uniformly inside the box
    randL = wf.L*np.random.rand(nconfig)
    #comput vgl, eloc along the x axis only 
    initpos[:,1,0] = randL
    newpos = np.zeros(initpos.shape)
    newpos[:,0,1] = np.ones(newpos.shape[0]) #modifies y coordinate of 1st electron
    newpos = initpos + newpos
    config = wf.setup(initpos)
    val = wf.val(initpos)
    #valb = wf.val(newpos) 
    
    #ratio = wf.valnew(newpos) ** 2 / wf.valnew(initpos) ** 2
    #print(ratio)

    g = wf.grad(initpos)[:,0,0] #pull out positive x derivative
    lap = wf.lap(initpos)[0]

    # use hacked energy in estimator
    if ((config.configs == initpos).all() == False) or ((wf._configscurrent.configs == newpos).all() == False):
        print('oogly boogly')
    eloc = GetEnergy(wf,config,initpos)
    xs = np.sqrt(np.sum((config.configs[:,0,:]-config.configs[:,1,:])**2,axis=1))
    fig, ax = plt.subplots(2, 2, sharex=True) 
    ax[0,0].plot(xs,val,'b.',label='value')
    ax[0,1].plot(xs,g,'r.',label='gradient')
    ax[1,0].plot(xs,lap,'g.',label='laplacian')
    ax[1,1].plot(xs,eloc,'k.',label='eloc')
    ax[0,0].axvline(wf.L/2)
    ax[0,1].axvline(wf.L/2)
    ax[1,1].axvline(wf.L/2)
    ax[1,0].axvline(wf.L/2)
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
    rs = sys.argv[1]
    print(rs)
    wf = UpdatedJastrow(rs)
    plotVGL(wf)
