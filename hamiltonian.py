
import numpy as np


class Hamiltonian:
  def __init__(self, g,hw,U=2.):
    '''
    Inputs:
        U: Coulomb strength
        g: density of states = 2hbar*w sqrt(pi*alpha*l/V)
        hw: characteristic phonon energy, in units of Bohr radius a0 --> hw = (a0/l)^2
    '''
    self.U = U
    self.g = g
    self.hw = hw

  def pot_ee(self,pos):
    """ electron-electron potential of configurations 'pos'. Need to implement Ewald sum """
    r12 = np.linalg.norm(pos[0,:,:]-pos[1,:,:],axis=0)
    return self.U/r12

if __name__=="__main__":
  # This part of the code will test your implementation. 
  # Don't modify it!
  pos=np.array([[[0.1],[0.2],[0.3]],[[0.2],[-0.1],[-0.2]]]) #r1 and r2
  f_ks = np.array([[0.-2.j], [0.-1.33333333j]])
  ham=Hamiltonian(U=4,g=2, hw=0.5)
  print("Error:")
  print(ham.pot_ph(f_ks) - -12.0118915)
  print(ham.pot_ee(pos) -  1.69030851)
  print(ham.pot(pos, f_ks) - -10.321583)
