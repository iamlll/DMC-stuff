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
        r'$b=%.4f \pm %.4f$' % (b, berr)
        ))

    print(r'$b=%.4f \pm %.4f$' % (b, berr))
    return ans, textstr

def PlotErr(df, xvar='tau', yvar='eavg',err='err', units='ha', fit=True):
    '''
    Plot E vs tau (timestep) from the reblocked DMC results in Ry (Rydberg) or ha (Hartree)
    '''

    #if multiple files, split by r_s value
    rsarr = np.unique(df['r_s'].values)
    print(rsarr)
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

        ax.plot(taus, Es, 'r.',label='nconfig = 500')
        ax.errorbar(taus, Es, yerr = yerr, fmt='r.')
        
        if r_s == 2:
            #Paul's results using uniform WF, r_s=2
            #576 walkers
            dts = [0.2, 0.1, 0.05]
            E500 = [-0.717854, -0.717702, -0.714989]
            err500 = [0.000739, 0.000845, 0.000492]
            #1536 walkers
            E1500 = [-0.717587, -0.717096, -0.716748]
            err1500 = [0.000425, 0.000419, 0.000465]

            ax.plot(dts, E500, 'g.',label='nconfig = 576')
            ax.errorbar(dts, E500, yerr = err500, fmt='g.')
            ax.plot(dts, E1500, 'b.',label='nconfig = 1536')
            ax.errorbar(dts, E1500, yerr = err1500, fmt='b.')
      
        if fit == True:
            extrap_x = taus
            #extrap_x = np.linspace(0,0.2,30)
            f1, t1 = FitData(taus,Es, yerr, extrap=extrap_x)
            ax.plot(extrap_x, f1, 'r')
            if r_s == 2:
                f2, t2 = FitData(dts,E500, err500,extrap=extrap_x)
                ax.plot(extrap_x, f2, 'g')
                f3, t3 = FitData(dts,E1500, err1500, extrap=extrap_x)
                ax.plot(extrap_x, f3, 'b')
        
            ax.text(0.05, 0.75, t1, transform=ax.transAxes, fontsize=14, verticalalignment='top')

        ax.set_xlabel('$\\tau$ (1/ha)')
        ax.set_ylabel('$E$ (ha)')
        ax.legend()
        plt.tight_layout()
        plt.show()
        #plt.savefig("traces.pdf", bbox_inches='tight')
    
if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    #PlotVars(df,yvars=['ke','pot','elocal']) 
    #PlotVars(df,yvars=['ke','pot','elocal','acceptance']) 
    PlotErr(df)
