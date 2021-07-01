import numpy as np
import sys
sys.path.append("../pyqmc")
from pyqmc import ewald, coord
from pyscf.pbc import gto

class Hamiltonian:
  def __init__(self, g,hw,U=2., Z=2):
    '''
    Inputs:
        U: Coulomb strength
        g: density of states = 2hbar*w sqrt(pi*alpha*l/V)
        hw: characteristic phonon energy, in units of Bohr radius a0 --> hw = (a0/l)^2
    '''
    self.U = U
    self.g = g
    self.hw = hw
    self.Z=Z
    
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

  def ewald(self, pos, L):
    '''Need to get pos into (nconf, nelec, ndim) shape; currently elec positions have shape (nelec,ndim,nconf) - should move this from hamiltonian to DMC file
    input: 
      L: system size
      ewaldobj: Ewald class object (from PyQMC)
    '''
    #create supercell
    cell = gto.Cell() #see PySCF docs for details
    cell.build(atom='He 0 0 0', basis='gth-dzvp', a=np.eye(pos.shape[1])*L, unit='B') #B = units of Bohr radii, I don't know what this choice of basis is but I don't think it matters, none of the Ewald functions seem to call on it
    #create Ewald object instance
    ewaldobj = ewald.Ewald(cell)
    
    swappos = np.moveaxis(pos,-1,0)
    #create PeriodicConfigs object
    pbcconfigs = coord.PeriodicConfigs(swappos, ewaldobj.latvec) 
    ee, ei, ii = ewaldobj.energy(pbcconfigs)
    coul = ee + ei + ii
    return self.U * coul

  def pot_ewald(self,pos, L=5):
    """ potential energy of configuations 'pos' """
    return 0.5*( self.pot_en(pos)+self.ewald(pos, L) )

if __name__=="__main__":
  # This part of the code will test your implementation. 
  # Don't modify it!
  np.random.seed(0) 
  pos=np.random.randn(2,3,1) #2x3x5 dimensional array of random numbers plucked from a normal distn (mean = 0, stdev = 1); [nelec, ndim, nconfig]
  f_ks = np.array([[0.-2.j], [0.-1.33333333j]])
  ham=Hamiltonian(U=4,g=2, hw=0.5)
  
  print(ham.pot_ee(pos))
  print(ham.ewald(pos, 5))
  print(ham.ewald(pos, 50))
  print(ham.ewald(pos, 100))
  print(ham.ewald(pos, 1000))
