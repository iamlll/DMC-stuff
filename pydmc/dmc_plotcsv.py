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
    df["step"] += 1
    taus = np.unique(df['tau'].values)
    for tau in taus:
        df2 = df[df['tau']==tau]
        print(df.keys())
        #g=sns.PairGrid(df2,x_vars=xvar, y_vars=yvars,hue='popstep')
        g=sns.PairGrid(df2,x_vars=xvar, y_vars=yvars)
        nequil = int(t_equil/tau)
        plt.gca().axvline(nequil) #plot how many steps thrown out during reblocking procedure
        g.map(plt.scatter,s=1)
        g.add_legend()
        plt.tight_layout()
        plt.show()
    #plt.savefig("traces.pdf", bbox_inches='tight')

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
    df = pd.read_csv(sys.argv[1])
    PlotErr(df,yvar='eavg',err='err')
