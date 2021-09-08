import numpy as np
import sys
sys.path.append("../pyqmc")
from pyqmc import ewald, coord
from pyscf.pbc import gto

class Hamiltonian:
  def __init__(self, U=2., Z=2, L=10):
    '''
    Inputs:
        U: Coulomb strength
        Z: nuclear charge
        L: supercell length
    '''
    self.U = U
    self.Z=Z
    self.L = L
    
    ndim = 3
    #create supercell
    cell = gto.Cell() #see PySCF docs for details
    cell.build(atom='He 0 0 0', a=np.eye(ndim)*self.L, unit='B') #B = units of Bohr radii
    #create Ewald object instance
    self.ewaldobj = ewald.Ewald(cell)
    
  def pot_en(self,pos):
    """ electron-nuclear potential of configurations 'pos' """
    r=np.sqrt(np.sum(pos**2,axis=1))
    return np.sum(-2*self.Z/r,axis=0)

  def pot_ee(self,pos):
    """ electron-electron potential of configurations 'pos'. """
    r12 = np.linalg.norm(pos[0,:,:]-pos[1,:,:],axis=0)
    return self.U/r12

  def pot(self,pos):
    """ potential energy of configuations 'pos' """
    return 0.5*( self.pot_en(pos)+self.pot_ee(pos) )

  def ewald(self, pos):
    '''Need to get pos into (nconf, nelec, ndim) shape; currently elec positions have shape (nelec,ndim,nconf)
    input: 
      ewaldobj: Ewald class object (from PyQMC)
    '''
    swappos = np.moveaxis(pos,-1,0)
    #create PeriodicConfigs object
    pbcconfigs = coord.PeriodicConfigs(swappos, self.ewaldobj.latvec) 
    ee, _,_ = self.ewaldobj.energy(pbcconfigs)
    return self.U*ee #ignore e-ion and ion-ion contributions

if __name__=="__main__":
  # This part of the code will test your implementation. 
  # Don't modify it!
  np.random.seed(0) 
  pos=np.random.randn(2,3,1) #2x3x5 dimensional array of random numbers plucked from a normal distn (mean = 0, stdev = 1); [nelec, ndim, nconfig]
  
  f_ks = np.array([[0.-2.j], [0.-1.33333333j]])
  ham=Hamiltonian(L=100)
  
  print(ham.pot_ee(pos))
  print(ham.ewald(pos))
  ham=Hamiltonian(L=1000)
  print(ham.ewald(pos))
  ham=Hamiltonian(L=100000)
  print(ham.ewald(pos))
  ham=Hamiltonian(L=1E6)
  print(ham.ewald(pos))
