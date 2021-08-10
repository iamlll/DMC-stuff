#!/usr/bin/env python3
import numpy as np
import sys
sys.path.append("../pyqmc")
from pyqmc import coord

class Hamiltonian:
  def __init__(self, L, **kwargs):
    from pyscf.pbc import gto
    from pyqmc import ewald
    self.e2byeps = 1  # Coulomb interaction strength e^2/epsilon
    #create supercell
    cell = gto.Cell() #see PySCF docs for details
    ndim = pos.shape[1]
    cell.build(atom='He 0 0 0', a=np.eye(ndim)*L, unit='B')
    #create Ewald object instance
    self.ewaldobj = ewald.Ewald(cell, **kwargs)
  def ewald(self, pos):
    swappos = np.moveaxis(pos,-1,0)
    #create PeriodicConfigs object
    pbcconfigs = coord.PeriodicConfigs(swappos, self.ewaldobj.latvec) 
    ee, _, _ = self.ewaldobj.energy(pbcconfigs)
    return self.e2byeps*ee

if __name__ == '__main__':
  nconf = 10000
  #wf = UniformWF()  # no need to run VMC for uniform wf
  rsl = [1, 2, 4, 8]
  for rs in rsl:
    lbox = (4*np.pi*2/3)**(1/3) * rs
    pos = lbox*np.random.randn(2, 3, nconf)
    ham = Hamiltonian(lbox, ewald_gmax=200, nlatvec=1)  # default
    #ham = Hamiltonian(lbox, ewald_gmax=300, nlatvec=3)
    el = ham.ewald(pos)

    em = np.mean(el)
    ee = np.std(el, ddof=1)/len(el)**0.5
    print(rs, em.round(3), ee.round(3))
# end __main__
