import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
from scipy.optimize import curve_fit

t_equil = 20
sns.set_style("white")
def PlotVars(df, xvar=['step'], yvars=['elocal']):
    '''
    Plot variables from the direct DMC results. If want to plot multiple y variables, input should be as an array
    '''
    #df["step"] += 1
    taus = np.unique(df['tau'].values)
    fig,ax = plt.subplots(2,1,figsize=(6,4.5),sharex='row')
    r_s = df['r_s'].values[0]
    print(r_s)
    for tau in taus:
        df2 = df[df['tau']==tau]
        print(df.keys())
        steps = df2[xvar].values
        Elocals = df2[yvars].values
        eloc = np.array([complex(val) for val in Elocals]) 
        ax[0].plot(steps,eloc.real,'k',label='real')
        ax[1].plot(steps,eloc.imag,'r',label='imag')
        plt.title('$r_s = $' + str(r_s))
        #g=sns.PairGrid(df2,x_vars=xvar, y_vars=yvars,hue='popstep')
        #g=sns.PairGrid(df2,x_vars=xvar, y_vars=yvars)
        nequil = int(t_equil/tau)
        ax[0].axvline(nequil) #plot how many steps thrown out during reblocking procedure
        ax[1].axvline(nequil) #plot how many steps thrown out during reblocking procedure
        ax[0].legend()
        ax[1].legend()
        ax[1].set_xlabel('time step')
        ax[0].set_ylabel('Elocal')
        ax[1].set_ylabel('Elocal')
        plt.tight_layout()
        plt.show()
    #plt.savefig("traces.pdf", bbox_inches='tight')

def RandTrials(filenames):
    '''
    Find avg energy + variance as a function of time step for a single r_s value(generated multiple sims using different random seeds, and want to determine when the calculation has converged wrt tau + what that converged answer is)
    Then plot this averaged answer
    '''
    def reblock(eloc,warmup,nblocks):
        elocblock=np.array_split(eloc[warmup:],nblocks) #throw out "warmup" number of equilibration steps and split resulting local energy array into nblocks subarrays
        print(tau,len(elocblock))
        blockenergy=[np.mean(x) for x in elocblock]
        return np.mean(blockenergy),np.std(blockenergy)/np.sqrt(nblocks)
    df0 = pd.read_csv(filenames[0])
    steps = df0['step'].values
    nsteps = len(steps)
    print(nsteps)
    r_s = df0['r_s'].values[0]
    tau = df0['tau'].values[0]
    Elocs = np.empty((len(filenames),nsteps),dtype=complex)
    for i,name in enumerate(filenames):
        df=pd.read_csv(name)
        for tau,grp in df.groupby("tau"):
            eloc=grp.sort_values('step')['elocal'].values
            Elocs[i,:] = np.array([complex(val) for val in eloc]) 
            #nequil = int(t_equil/tau)
            #nblocks=int((len(eloc)-nequil)/blocktau)
            #avg,err=reblock(eloc,nequil,nblocks)
    avgE = np.mean(Elocs,axis=0)
    SD_real = np.std(Elocs.real,axis=0)/np.sqrt(len(filenames)-1)
    SD_imag = np.std(Elocs.imag,axis=0)/np.sqrt(len(filenames)-1) #numpy only returns a real value for standard deviation
    print(np.mean(avgE[150:]))
    print(np.mean(SD_real[150:]))
    print(np.mean(SD_imag[150:]))
    fig,ax = plt.subplots(2,1,figsize=(6,4.5),sharex='row')
    ax[0].errorbar(steps,avgE.real,yerr=SD_real,label='real')
    ax[0].plot(steps,avgE.real,'r',zorder=10)
    ax[1].errorbar(steps,avgE.imag,yerr=SD_imag,label='imag')
    ax[1].plot(steps,avgE.imag,'r',zorder=10)
    ax[0].legend()
    ax[1].legend()
    ax[1].set_xlabel('time step')
    ax[0].set_ylabel('Re[Elocal]')
    ax[1].set_ylabel('Im[Elocal]')
    plt.title('$r_s = $' + str(r_s) + ', $\\tau = $' + str(tau))
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
        r'$b=%.5f \pm %.5f$' % (b, berr)
        ))

    print(r'$b=%.5f \pm %.5f$' % (b, berr))
    return ans, textstr

def PlotErr(df, xvar='tau', yvar='eavg',err='err', units='ha', fit=True):
    '''
    Plot E vs tau (timestep) from the reblocked DMC results in Ry (Rydberg) or ha (Hartree)
    '''

    #if multiple files, split by r_s value
    #nconfig = df['nconfig'].values[0]
    nconfig=512
    rsarr = np.unique(df['r_s'].values)
    for r_s in rsarr:
        print(r_s)
        fig = plt.figure(figsize=(6,4.5))
        ax = fig.add_subplot(111)
        dfnew = df[df['r_s']==r_s] 
        taus = dfnew[xvar].values
        Es = dfnew[yvar].values
        yerr = dfnew[err].values
        if units == 'Ry':
            Es = Es*4 #convert from total ha to Ry
        print(taus) 
        print(Es)
        print(yerr)
        ax.plot(taus, Es, 'r.',label='$r_s = $' + str(r_s) + ', nconfig = ' + str(nconfig))
        ax.errorbar(taus, Es, yerr = yerr, fmt='r.')
        
        if r_s == 4:
            dts = [0.1,0.2,0.4]
            E1k = [-0.365,-0.365,-0.364]
            err1k = [0.003,0.002,0.0019]

            ax.plot(dts, E1k, 'g.',label='PyQMC driver, nconfig = 1000')
            ax.errorbar(dts, E1k, yerr = err1k, fmt='g.')
        elif r_s == 2:
            dts = [0.1,0.2,0.4]
            E1k = [-0.715,-0.717,-0.717]
            err1k = [0.004,0.004,0.003]

            ax.plot(dts, E1k, 'g.',label='PyQMC driver, nconfig = 1000')
            ax.errorbar(dts, E1k, yerr = err1k, fmt='g.')
        elif r_s == 1:
            dts = [0.1,0.2,0.4]
            E1k = [-1.413,-1.419,-1.418]
            err1k = [0.008,0.005,0.006]

            ax.plot(dts, E1k, 'g.',label='PyQMC driver, nconfig = 1000')
            ax.errorbar(dts, E1k, yerr = err1k, fmt='g.')
        elif r_s == 8:
            dts = [0.1,0.2,0.4]
            E1k = [-0.1865,-0.1864,-0.1862]
            err1k = [0.0017,0.0019,0.0014]

            ax.plot(dts, E1k, 'g.',label='PyQMC driver, nconfig = 1000')
            ax.errorbar(dts, E1k, yerr = err1k, fmt='g.')
      
        if fit == True:
            extrap_x = taus
            #extrap_x = np.linspace(0,0.2,30)
            f1, t1 = FitData(taus,Es, yerr, extrap=extrap_x)
            ax.plot(extrap_x, f1, 'r')
            if r_s == 4 or r_s == 1 or r_s == 2 or r_s == 8:
                f2, t2 = FitData(dts,E1k, err1k,extrap=extrap_x)
                ax.plot(extrap_x, f2, 'g')
        
            ax.text(0.05, 0.3, t1, transform=ax.transAxes, fontsize=14, verticalalignment='top')

        ax.set_xlabel('$\\tau$ (1/ha)')
        ax.set_ylabel('$E$ (ha)')
       
        ax.legend()
        plt.tight_layout()
        plt.show()
        #plt.savefig("traces.pdf", bbox_inches='tight')
    
def CompareExtrapKE():
    t1 = np.array([0.0125,0.025,0.05,0.1])
    K1_DMC = np.array([0.00062,0.00038,0.00021,0.00012])
    err1_DMC = np.array([3E-5,4E-5,4E-5,6E-5])
    t4 = np.array([0.05,0.1,0.2,0.4])
    K4_DMC = np.array([0.000767,0.000650,0.000519,0.000328])
    err4_DMC = np.array([1.8E-5,2E-5,3E-5,4E-5])
    t10 = np.array([0.125,0.25,0.5,1.])
    K10_DMC = np.array([0.000467,0.000439,0.000404,0.00034])
    err10_DMC = np.array([8E-6,8E-6,1.1E-5,2E-5])
    K1_VMC = np.array([0.00006,5E-5,8E-5,3E-5])
    err1_VMC = np.array([2E-5,2E-5,3E-5,4E-5])
    K4_VMC = np.array([0.00011,0.00011,0.00009,0.00013])
    err4_VMC = np.array([3E-5,3E-5,3E-5,3E-5])
    K10_VMC = np.array([0.000131,0.000126,0.000111,0.00013])
    err10_VMC = np.array([1.8E-5,1.8E-5,1.9E-5,2E-5])
    #pull out 2*DMC-VMC kinetic energies for different rs values
    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(111)
    ax.errorbar(t1,2*K1_DMC-K1_VMC,fmt='o',yerr=2*err1_DMC-err1_VMC,label='$r_s=1$')
    ax.errorbar(t4,2*K4_DMC-K4_VMC,fmt='o',yerr=2*err4_DMC-err4_VMC, label='$r_s=4$')
    ax.errorbar(t10,2*K10_DMC-K10_VMC,fmt='o',yerr=2*err10_DMC-err10_VMC, label='$r_s=10$')
    ax.legend()
    ax.set_ylabel("$2T_{VMC}-T_{DMC}$ (ha)")
    ax.set_xlabel("$\\tau$ (1/ha)")
    plt.tight_layout()
    plt.show()    

if __name__ == "__main__":
    #df = pd.read_csv(sys.argv[1])
    #PlotErr(df,yvar='eavg',err='err')
    #PlotVars(df)
    RandTrials(sys.argv[1:])
