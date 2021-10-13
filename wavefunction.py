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
    print(lap_ee.shape)
    #Laplacian is the same for both electrons
    return np.array([lap_ee,lap_ee])

  #-------------------------

class PBCJastrowWF:
  """
  Jastrow factor of the form 
  exp(-u(r1,r2))

  u(r1,r2) = A/r12* (1-exp(-r12/F))
  r12 = |r_1 - r_2| 
  which obeys periodic boundary conditions (PBC) (unrestricted particle coordinates)
  #need to cut off u(r1,r2) at L/2 since it diverges there. Do so using a Taylor expansion
  """
  def __init__(self, L, opt, smooth, ncells=1):
    self.L = L
    self.units = 1. #convert from ha to Ry, 1 ha = 2 Ry
    if smooth == True: #Taylor expn t(r) != 0
          self.A = 1./np.sqrt(4*np.pi*2./L**3) #keep in mind this is in units of ha!!
          self.F = np.sqrt(self.A*2)
          self.A = 0.5* (1/self.F**2 + 4/self.L * (-2/self.L*(1-np.exp(-self.L/(2*self.F))) + 1./self.F*np.exp(-self.L/(2*self.F))))**(-1)
    else: 
        self.A = 1./np.sqrt(4*np.pi*2./L**3) #keep in mind this is in units of ha!!
        self.F = np.sqrt(self.A*2)
    self.M = ncells #number of simulation cells to sum over
    self.smooth = smooth #determines whether to smooth out u(r) at r = L/2 by subtracting off the Taylor expn of u(r) (which we will call t(r))
    self.constrainPBC = opt #constrain electrons to the physical box size L
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
    u = lambda x: self.A/x* (1-np.exp(-x/self.F))
    if self.smooth == True:
        dudr = lambda x: self.A/x * (-1./x * (1-np.exp(-x/self.F)) + 1./self.F*np.exp(-x/self.F) )
        d2udr2 = lambda x: self.A* (2/x**3 *(1-np.exp(-x/self.F)) - np.exp(-x/self.F)/(x*self.F) * (2/x + 1./self.F) )
        t = u(self.L/2) + dudr(self.L/2) * (eedist-self.L/2) + 0.5*d2udr2(self.L/2) * (eedist - self.L/2)**2
        return np.exp(-self.units* (u(eedist)-t) )
        #return u(eedist), t, u(eedist)-t #just use for Test_Jastrow fxn comparing smoothed wfn @ r=L/2 with unsmoothed/original
    else: return np.exp(-self.units*u(eedist))

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
    grad = self.A*dx/eedist**2 *(-1/eedist + (1/eedist + 1/self.F)*np.exp(-eedist/self.F))
    if self.smooth == True:
        dudr = lambda x: self.A/x * (-1./x * (1-np.exp(-x/self.F)) + 1./self.F*np.exp(-x/self.F) )
        d2udr2 = lambda x: self.A* (2/x**3 *(1-np.exp(-x/self.F)) - np.exp(-x/self.F)/(x*self.F) * (2/x + 1./self.F) )
        gradt = dudr(self.L/2) * dx/eedist + d2udr2(self.L/2) * (eedist - self.L/2) * dx/eedist
        pdee=np.outer([1,-1],-self.units*(grad-gradt) ).reshape(pos.shape)
    else:
        pdee=np.outer([1,-1],-self.units*grad).reshape(pos.shape)
    
    return pdee 
  #-------------------------

  def laplacian(self,pos): #WRONG
    if self.constrainPBC == True:
        pos = np.where(pos >= self.L, pos - self.L, pos)
        pos = np.where(pos < 0, pos + self.L, pos)
    #impose periodic boundary conditions (minimum image convention)
    dx = pos[0,:,:] - pos[1,:,:] #distance direction vector from electron 1 to electron 2; 3 x nconfig array
    dx = dx - self.L*np.rint(dx/self.L)

    eedist=(np.sum(dx**2,axis=0)**0.5)[np.newaxis,:]
    grad = self.A*dx/eedist**2 *(-1/eedist + (1/eedist + 1/self.F)*np.exp(-eedist/self.F))
    pdee=np.outer([1,-1],grad).reshape(pos.shape)
    pdee2=pdee[0]**2 # Sign doesn't matter if squared.
    pd2ee = self.A* (1./eedist**2*(-1./eedist *(1-np.exp(-eedist/self.F)) + 1./self.F*np.exp(-eedist/self.F)) + dx**2/eedist**2*(3/eedist**3 *(1-np.exp(-eedist/self.F)) - 3/(eedist**2 *self.F)*np.exp(-eedist/self.F) - np.exp(-eedist/self.F)/(eedist*self.F**2) ) )
    lap_ee= -self.units* np.sum(pd2ee - self.units*pdee2,axis=0)

    if self.smooth == True:
        dudr = lambda x: self.A/x * (-1./x * (1-np.exp(-x/self.F)) + 1./self.F*np.exp(-x/self.F) )
        d2udr2 = lambda x: self.A* (2/x**3 *(1-np.exp(-x/self.F)) - np.exp(-x/self.F)/(x*self.F) * (2/x + 1./self.F) )
        d3udr3 = lambda x: self.A* (-6/x**4 *(1-np.exp(-x/self.F)) + np.exp(-x/self.F) * (6/(x**3*self.F) + 3/(x**2*self.F**2) + 1./(x* self.F**3)) )
        d4udr4 = lambda x: self.A* (24/x**5 *(1-np.exp(-x/self.F)) - np.exp(-x/self.F) * (24/(x**4*self.F) + 12/(x**3*self.F**2) + 4/(x**2* self.F**3) + 1./(x*self.F**4)) )
        lapt = 2/eedist*dudr(self.L/2) + d2udr2(self.L/2) * (3-self.L/eedist) + d3udr3(self.L/2) * (eedist-self.L/2)*(2-self.L/(2*eedist)) + d4udr4(self.L/2) * (eedist-self.L/2)**2 *(5/6 - self.L/(6*eedist)) 
        lap_ee = lap_ee - lapt[0][0]
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
    nabla2u = -self.A*np.exp(-eedist/self.F)/(eedist*self.F**2) 
    grad = self.A/eedist *(-1/eedist + (1/eedist + 1/self.F)*np.exp(-eedist/self.F))

    if self.smooth == True:
        dudr = lambda x: self.A/x * (-1./x * (1-np.exp(-x/self.F)) + 1./self.F*np.exp(-x/self.F) )
        d2udr2 = lambda x: self.A* (2/x**3 *(1-np.exp(-x/self.F)) - np.exp(-x/self.F)/(x*self.F) * (2/x + 1./self.F) )
        nabla2t = 2/eedist*dudr(self.L/2) + d2udr2(self.L/2) * (3-self.L/eedist)
        gradt = dudr(self.L/2) + d2udr2(self.L/2) * (eedist - self.L/2) 
        lap_ee = -(nabla2u - nabla2t - (grad-gradt)**2)
    else: lap_ee = -(nabla2u - grad**2)
    return np.array([lap_ee,lap_ee])
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
  print(wf.laplacian(testpos))
  print(wf.nabla2(testpos))
  
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
   
def test_wavefunction(wf,L):
  """ test """
  #testpos=np.random.randn(2,3,5)
  testpos=np.zeros((2,3,1))
  testpos[1,:,0] = [L/16,0,0] #modify 1st walker, 1st electron's position
  testpos[0,:,0] = [0,0.2,-0.3] #modify 1st walker, 1st electron's position
  #testpos[1,:,0] = [1.7,-0.5,1.0] #modify 1st walker, 2nd electron's position
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

def Test_Jastrow(wf, nconfig=5):
    import matplotlib.pyplot as plt

    initpos = np.zeros((2,3,nconfig))
    xs = (wf.L)*np.random.rand(nconfig)

    initpos[1,0,:] = xs
    u,t,diff = wf.value(initpos)
    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(111)
    ax.plot(xs, u, '.',label='u(r)')
    ax.plot(xs, t, '.',label='t(r)')
    ax.plot(xs, diff, '.',label='u(r)-t(r)')
    ax.set_xlabel('x')
    ax.legend()
    plt.tight_layout()
    plt.show()

########################################

if __name__=="__main__":
  import pandas as pd
  testpos=np.zeros((2,3,1))
  #print("Uniform wavefunction")
  #uni=UniformWF()
  #test_wavefunction(uni)

  print("Jastrow wavefunction")
  L = 5
  jas=PBCJastrowWF(L,True,True)
  #jas=JastrowWF(0.5)
  #test_wavefunction(jas, L)
  Test_Jastrow(jas, 1000)
  #jas2=JastrowWF(-1.0)
  #test_wavefunction(jas2)

  #print("Multiplied wavefunction")
  #mwf=MultiplyWF(JastrowWF(1.0),JastrowWF(0.8))
  #test_wavefunction(mwf)

