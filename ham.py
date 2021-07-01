import numpy as np
import sys
sys.path.append("../pyqmc")
from pyqmc import ewald
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
    """ electron-electron potential of configurations 'pos'. Need to implement Ewald sum """
    r12 = np.linalg.norm(pos[0,:,:]-pos[1,:,:],axis=0)
    return self.U/r12

  def pot(self,pos):
    """ potential energy of configuations 'pos' """
    return 0.5*( self.pot_en(pos)+self.pot_ee(pos) )

  def ewald(self, pos, L):
    '''Need to get pos into (nconf, nelec, ndim) shape; currently elec positions have shape (nelec,ndim,nconf) - should move this from hamiltonian to DMC file
    input: 
      L: system size
    '''
    swappos = np.moveaxis(pos,-1,0)
    #create supercell
    cell = gto.Cell() #see PySCF docs for details
    cell.build(atom='He 0 0 0', basis='gth-dzvp', a=np.eye(swappos.shape[1])*L) #I don't know what this choice of basis is
    #create Ewald object instance
    ewaldobj = ewald.Ewald(cell)
    print(ewaldobj)   
    ee, ei, ii = ewaldobj.energy(swappos)
    coul = ee + ei + ii
    print(coul)
    return ham.U * np.moveaxis(coul, 0,-1) #get back into original position config; 0.5 to get into natural units; get rid of this when re-inserting phonons

  def pot_ewald(self,pos, L=5):
    """ potential energy of configuations 'pos' """
    return 0.5*( self.pot_en(pos)+self.ewald(pos, L) )

if __name__=="__main__":
  # This part of the code will test your implementation. 
  # Don't modify it!
  pos=np.array([[[0.1,0.2,0.3]],[[0.2,-0.1,-0.2]]]) #r1 and r2
  print(pos)
  f_ks = np.array([[0.-2.j], [0.-1.33333333j]])
  ham=Hamiltonian(U=4,g=2, hw=0.5)
  #print(ham.g)
  #print("Error:")
  print(ham.pot_ewald(pos))
