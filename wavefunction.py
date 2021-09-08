import numpy as np
import matplotlib.pyplot as plt

class UniformWF:
  '''
  A uniform wave function fixed at the value 1
  '''
  def __init__(self,val=1.):
    self.val=val
  def value(self, pos):
    scalar = np.full(pos.shape[-1], self.val)
    return scalar
  def gradient(self, pos):
    grad = np.zeros(pos.shape)
    return grad
  def laplacian(self, pos):
    scalar = pos.sum(axis=1)
    scalar.fill(0)
    return scalar

class JastrowWF:
  """
  Jastrow factor of the form 

  exp(J_ee)

  J_ee = a_ee|r_1 - r_2| 

  """
  def __init__(self,a_ee):
    self.a_ee=a_ee
  #-------------------------

  def value(self,pos):
    eedist=np.sqrt(np.sum((pos[0,:,:]-pos[1,:,:])**2,axis=0))
    exp_ee=(self.a_ee*eedist) #/(1 + self.eep_den*eedist)
    return np.exp(exp_ee)

  #-------------------------

  def gradient(self,pos):
    eedist=(np.sum((pos[0,:,:]-pos[1,:,:])**2,axis=0)**0.5)[np.newaxis,:] #newaxis turns the 1D array -> 2D array and copies the 1D array values over to the new axis
    # Partial derivatives of electron-electron distance, i.e. Jacobian
    #Outer product: (a0 * (b0 b1 b2) = (a0b0 a0b1 a0b2
    #                a1)                a1b0 a1b1 a1b2)
    #is the product of an (Mx1) and (1xN) array (e.g. above, M = 2, N = 3)
    pdee=np.outer([1,-1],(pos[0,:,:]-pos[1,:,:])/eedist).reshape(pos.shape)
    grad_ee=self.a_ee*pdee 
    return grad_ee 
  #-------------------------

  def laplacian(self,pos):
    eedist=(np.sum((pos[0,:,:]-pos[1,:,:])**2,axis=0)**0.5)[np.newaxis,:]
    pdee=np.outer([1,-1],(pos[0,:,:]-pos[1,:,:])/eedist).reshape(pos.shape)
    pdee2=pdee[0]**2 # Sign doesn't matter if squared.
    pd2ee=(eedist**2-(pos[0,:,:]-pos[1,:,:])**2)/eedist**3
    lap_ee=np.sum(self.a_ee*pd2ee + self.a_ee**2*pdee2,axis=0)
    #Laplacian is the same for both electrons
    return np.array([lap_ee,lap_ee])

  #-------------------------

class PBCJastrowWF:
  """
  Jastrow factor of the form 
  exp(J_ee)

  J_ee = A/r12* (1-exp(-r12/F))
  r12 = |r_1 - r_2| 
  which obeys periodic boundary conditions (PBC) (unrestricted particle coordinates)

  """
  def __init__(self,a_ee, L, opt):
    self.a_ee=a_ee
    self.L = L
    self.A = 1./np.sqrt(4*np.pi*2./L**3)
    self.constrainPBC = opt
  #-------------------------

  def value(self,pos):
    '''dx: vector distance between electron 1 and closest image charge'''
    if self.constrainPBC == True:
        pos = np.where(pos >= self.L, pos - self.L, pos)
        pos = np.where(pos < 0, pos + self.L, pos)
    #impose periodic boundary conditions (minimum image convention)
    dx = pos[0,:,:] - pos[1,:,:] #distance direction vector from electron 1 to electron 2; 3 x nconfig array
    dx = dx - self.L*np.rint(dx/self.L)

    eedist=np.sqrt(np.sum(dx**2,axis=0)) #summing over x,y,z directions
    u = self.A/eedist* (1-np.exp(-eedist/self.A))
    return np.exp(u)

  #-------------------------

  def gradient(self,pos):
    if self.constrainPBC == True:
        pos = np.where(pos >= self.L, pos - self.L, pos)
        pos = np.where(pos < 0, pos + self.L, pos)
    #impose periodic boundary conditions (minimum image convention)
    dx = pos[0,:,:] - pos[1,:,:] #distance direction vector from electron 1 to electron 2; 3 x nconfig array
    dx = dx - self.L*np.rint(dx/self.L)

    eedist=(np.sum(dx**2,axis=0)**0.5)[np.newaxis,:] #newaxis turns the 1D array -> 2D array and copies the 1D array values over to the new axis
    # Partial derivatives of electron-electron distance, i.e. Jacobian
    #Outer product: (a0 * (b0 b1 b2) = (a0b0 a0b1 a0b2
    #                a1)                a1b0 a1b1 a1b2)
    #is the product of an (Mx1) and (1xN) array (e.g. above, M = 2, N = 3)
    grad = self.A*dx/eedist**2 *(-1/eedist + (1/eedist + 1/self.A)*np.exp(-eedist/self.A))
    pdee=np.outer([1,-1],grad).reshape(pos.shape)
    return pdee 
  #-------------------------

  def laplacian(self,pos):
    if self.constrainPBC == True:
        pos = np.where(pos >= self.L, pos - self.L, pos)
        pos = np.where(pos < 0, pos + self.L, pos)
    #impose periodic boundary conditions (minimum image convention)
    dx = pos[0,:,:] - pos[1,:,:] #distance direction vector from electron 1 to electron 2; 3 x nconfig array
    dx = dx - self.L*np.rint(dx/self.L)

    eedist=(np.sum(dx**2,axis=0)**0.5)[np.newaxis,:]
    pdee=np.outer([1,-1],dx/eedist).reshape(pos.shape)
    pdee2=pdee[0]**2 # Sign doesn't matter if squared.
    pd2ee=(eedist**2-dx**2)/eedist**3
    lap_ee=np.sum(self.a_ee*pd2ee + self.a_ee**2*pdee2,axis=0)
    #Laplacian is the same for both electrons
    return np.array([lap_ee,lap_ee])
  #-------------------------

  def nabla2(self,pos):
    if self.constrainPBC == True:
        pos = np.where(pos >= self.L, pos - self.L, pos)
        pos = np.where(pos < 0, pos + self.L, pos)
    #impose periodic boundary conditions (minimum image convention)
    dx = pos[0,:,:] - pos[1,:,:] #distance direction vector from electron 1 to electron 2; 3 x nconfig array
    dx = dx - self.L*np.rint(dx/self.L)

    eedist=np.sum(dx**2,axis=0)**0.5 #1 x nconfig array
    nabla2u = self.A* (2/eedist**3 - (2/eedist**3 + 2/(eedist**2*self.A) + 1./(eedist*self.A**2) )* np.exp(-eedist/self.A) )
    nablau2 = self.A**2/eedist**2 *(-1/eedist + (1/eedist + 1/self.A)*np.exp(-eedist/self.A))**2
    lap_ee = nabla2u + nablau2
    #Laplacian is the same for both electrons
    return np.array([lap_ee, lap_ee])

  #-------------------------
########################################

class MultiplyWF:
  """ Wavefunction defined as the product of two other wavefunctions."""
  def __init__(self,wf1,wf2):
    self.wf1=wf1
    self.wf2=wf2
#-------------------------
  def value(self,pos):
    return self.wf1.value(pos)*self.wf2.value(pos)
#-------------------------
  def gradient(self,pos):
    return self.wf1.gradient(pos) + self.wf2.gradient(pos)
#-------------------------
  def laplacian(self,pos):
    return self.wf1.laplacian(pos) +\
           2*np.sum(self.wf1.gradient(pos)*self.wf2.gradient(pos),axis=1) +\
           self.wf2.laplacian(pos)
#-------------------------

########################################

def derivative_test(testpos,wf,delta=1e-4):
  """ Compare numerical and analytic derivatives. """
  wf0=wf.value(testpos)
  grad0=wf.gradient(testpos)
  npart=testpos.shape[0]
  ndim=testpos.shape[1]
  grad_numeric=np.zeros(grad0.shape)
  for p in range(npart):
    for d in range(ndim):
      shift=np.zeros(testpos.shape)
      shift[p,d,:]+=delta
      wfval=wf.value(testpos+shift)
      grad_numeric[p,d,:]=(wfval-wf0)/(wf0*delta)
  
  return np.sqrt(np.sum((grad_numeric-grad0)**2)/(npart*testpos.shape[2]*ndim))

########################################
    
def laplacian_test(testpos,wf,delta=1e-5):
  """ Compare numerical and analytic Laplacians. """
  wf0=wf.value(testpos)
  lap0=wf.nabla2(testpos)
  npart=testpos.shape[0]
  ndim=testpos.shape[1]
  print(wf.laplacian(testpos)-wf.nabla2(testpos))
  
  lap_numeric=np.zeros(lap0.shape)
  for p in range(npart):
    for d in range(ndim):
      shift=np.zeros(testpos.shape)
      shift[p,d,:]+=delta      
      wf_plus=wf.value(testpos+shift)
      shift[p,d,:]-=2*delta      
      wf_minus=wf.value(testpos+shift)
      # Here we use the value so that the laplacian and gradient tests
      # are independent
      lap_numeric[p,:]+=(wf_plus+wf_minus-2*wf0)/(wf0*delta**2)
  
  return np.sqrt(np.sum((lap_numeric-lap0)**2)/(npart*testpos.shape[2]))

########################################
   
def test_wavefunction(wf):
  """ test """
  testpos=np.random.randn(2,3,5)
  df={'delta':[],
      'derivative err':[],
      'laplacian err':[]
      }
  for delta in [1e-2,1e-3,1e-4,1e-5,1e-6]:
    df['delta'].append(delta)
    df['derivative err'].append(derivative_test(testpos,wf,delta))
    df['laplacian err'].append(laplacian_test(testpos,wf,delta))

  import pandas as pd
  df=pd.DataFrame(df)
  print(df)
  return df

########################################

if __name__=="__main__":
  import pandas as pd
  testpos=np.random.randn(2,3,5)

  #print("Uniform wavefunction")
  #uni=UniformWF()
  #test_wavefunction(uni)

  print("Jastrow wavefunction")
  jas=PBCJastrowWF(-1.0,5,True)
  test_wavefunction(jas)
  #jas2=JastrowWF(-1.0)
  #test_wavefunction(jas2)

  #print("Multiplied wavefunction")
  #mwf=MultiplyWF(JastrowWF(1.0),JastrowWF(0.8))
  #test_wavefunction(mwf)

