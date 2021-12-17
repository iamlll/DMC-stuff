import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
from scipy.optimize import curve_fit

tequil = 20 #equilibration time = timestep * (# steps thrown out)
blocksize=1.0 # in Hartree-1

def reblocked(h5,colnames=['energyke','energytotal'],tequil=10,blocksize=1.0):
    def reblock(eloc,warmup,nblocks):
        elocblock=np.array_split(eloc[warmup:],nblocks) #throw out "warmup" number of equilibration steps and split resulting local energy array into nblocks subarrays
        blockenergy=[np.mean(x) for x in elocblock]
        return np.mean(blockenergy),np.std(blockenergy)/np.sqrt(nblocks)
    qtys = np.zeros(len(colnames))
    errs = np.zeros(len(colnames))
    tau = h5['tstep'][0]
    blocktau=blocksize/tau
    nequil = int(tequil/tau)
    for i,name in enumerate(colnames):
        data = np.sort(h5[name])
        nblocks=int((len(data)-nequil)/blocktau)
        avg,err=reblock(data,nequil,nblocks)
        qtys[i] = avg
        errs[i] = err
    return tau, qtys, errs     

def main(colnames=['energyke','energytotal']):
  filenames = sys.argv[1:]
  fig, ax = plt.subplots()
  taus = np.zeros(len(filenames))
  energies = np.zeros((len(colnames),len(filenames),))
  err = np.zeros((len(colnames),len(filenames),))
  for i,f in enumerate(filenames):
    h5 = h5py.File(f,'r')
    print(f)
    #trace = h5['energytotal']
    tstep = h5['tstep'][0]
    #tequil = 10
    #tcorr = 1
    #nskip = int(round(tequil/tstep))
    #nevery = int(round(tcorr/tstep))
    #trace1 = trace[nskip::nevery]
    #ym = np.mean(trace1)
    #ye = np.std(trace1, ddof=1)
    #print(ym)
    #print(ye)
    
    tstep, ym, ye = reblocked(h5,colnames) 
    energies[:,i] = ym
    err[:,i] = ye
    taus[i] = tstep
  print(energies)
  print(err)

  for j,name in enumerate(colnames):
    ax.errorbar(taus,energies[j,:],yerr=err[j,:],fmt='^',label=name)
    fit, _ = FitData(taus,energies[j,:])
    ax.plot(taus,fit)
  ax.legend()
  ax.set_xlabel('tau (1/ha)')
  ax.set_ylabel('tot. energy')
  plt.tight_layout()
  plt.show()

def FitData(xvals, yvals, yerr=[], fit='lin', extrap=[]):
    def fitlinear(x,a,b):
        f = a*x + b 
        return f

    bnds = ([-10,-10],[5,5]) #bounds for weak coupling fit
    guess =[-1,-3]
    if len(yerr) > 0:
        param, p_cov = curve_fit(fitlinear,xvals, yvals, sigma=yerr, p0=guess,bounds=bnds)
    else:
        param, p_cov = curve_fit(fitlinear,xvals, yvals, p0=guess,bounds=bnds)
    #print(param)
    a,b = param
    aerr, berr = np.sqrt(np.diag(p_cov)) #standard deviation of the parameters in the fit
    
    if len(extrap) > 0:
        ans = np.array([fitlinear(x,a,b) for x in extrap])
    else:    
        ans = np.array([fitlinear(x,a,b) for x in xvals])
    
    textstr = '\n'.join((
        r'$E(\tau) = a\tau + b$',
        r'$a=%.4f \pm %.4f$' % (a, aerr),
        r'$b=%.6f \pm %.6f$' % (b, berr)
        ))

    print(textstr)
    return ans, textstr

if __name__ == '__main__':
  main(['energytotal',])
