'''
Various accumulated plotting functions to visualize DMC data -- note that some of these require the raw dataset (i.e. for E vs time plots) while others require blocked data sets (which give the average energy over the whole run)
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
from scipy.optimize import curve_fit

t_equil = 20
sns.set_style("white")

def Emix_vs_gth(df):
    '''Plot growth vs mixed estimator results for the system energy, as a function of time'''
    fig,ax = plt.subplots(2,1,figsize=(6,4.5),sharex='row')
    fig.suptitle('$E_{mix}-E_{gth}$')
    r_s = df['r_s'].values[0]
    print(r_s)
    taus = np.unique(df['tau'].values)
    for tau in taus:
        df2 = df[df['tau']==tau]
        steps = df2['step'].values
        
        Earray1 = df2['elocal'].values
        Earray2 = df2['egth'].values
        eloc1 = np.array([complex(val) for val in Earray1]) 
        eloc2 = np.array([complex(val) for val in Earray2]) 
        ax[0].plot(steps,(eloc1-eloc2).real,label='$\\tau = $' + str(tau))
        ax[1].plot(steps,(eloc1-eloc2).imag,label='$\\tau = $' + str(tau))
        plt.title('$r_s = $' + str(r_s))
        nequil = int(t_equil/tau)
        ax[0].axvline(nequil,c='y') #plot how many steps thrown out during reblocking procedure
        ax[1].axvline(nequil,c='y') #plot how many steps thrown out during reblocking procedure
    ax[0].legend()
    ax[1].legend()
    ax[1].set_xlabel('time step')
    ax[0].set_ylabel('Re(E)')
    ax[1].set_ylabel('Im(E)')

    plt.tight_layout()
    plt.show()
        
def Extrapolate_Emix_gth(df):
    '''Extrapolate E_mix and E_gth as functions of the timestep tau to determine E_exact = E(tau->0). Then compare E_m(tau) and E_g(tau) with E_exact to see which one is closer'''
    fig,ax = plt.subplots(1,2,figsize=(8,4.5))
    r_s = df['r_s'].values[0]
    taus = df['tau'].values
    E_mix = df['eavg'].values
    err_mix = df['err'].values
    E_gth = df['egth'].values
    err_gth = df['err_gth'].values
    ax[0].errorbar(taus,E_mix,fmt='o',yerr=err_mix,label='mixed') 
    ax[0].errorbar(taus,E_gth,fmt='o',yerr=err_gth,label='growth')
    
    mixparam,fit_mix, txt_mix = FitData(taus,E_mix, yerr=err_mix, fit='lin',guess=[-1,1],varnames=['\\tau','E_{mix}'],retparam=True)
    fit_gth, txt_gth = FitData(taus,E_gth, yerr=err_gth, fit='lin',guess=[-1,1],varnames=['\\tau','E_{gth}'])
    ax[0].plot(taus,fit_mix,label='mixed fit')
    ax[0].plot(taus,fit_gth,label='growth fit')
    ax[0].text(0.6, 0.4, txt_mix, horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)
    ax[0].text(0.6, 0.6, txt_gth, horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)
    ax[0].set_xlabel('$\\tau$')
    ax[0].set_ylabel('$E$')
    ax[0].legend()
    #now plot difference between E_0 and E_mix, E_gth
    E0 = mixparam[1]
    ax[1].plot(taus,E0-E_mix,'.',label='mixed')
    ax[1].plot(taus,E0-E_gth,'.',label='growth')
    ax[1].legend()
    ax[1].set_xlabel('$\\tau$')
    ax[1].set_ylabel('$E_0-E$')
    plt.tight_layout()
    plt.show() 

def Timeplot_phonon_amps(df):
    '''Plot phonon amplitudes f_k as a function of sim step'''
    fig,ax = plt.subplots(2,1,figsize=(6,4.5),sharex='row')
    r_s = df['r_s'].values[0]
    print(r_s)
    taus = np.unique(df['tau'].values)
    for tau in taus:
        df2 = df[df['tau']==tau]
        steps = df2['step'].values
        f_ks = df2['f_ks'].values # (No timesteps) x (No kvectors) matrix
        ks = df2['ks'].values[0] # k vector magnitudes
        nparr = np.zeros((len(f_ks),len(f_ks[0])),dtype=complex)
        print(nparr.shape)
        for i in range(len(f_ks)):
            nparr[i,:] = f_ks[i]
        #pull out phonon amp corresponding to 1st kvec
        kidxs = [0,50,100,300]
        print(ks[kidxs])
        for idx in kidxs:
            f_k1 = nparr[:,idx] 
            ax[0].plot(steps,f_k1.real,label='$\\tau = %.2f,\, |\\vec k_{%d}|=%.2f$' %(tau, idx,ks[idx]) )
            ax[1].plot(steps,f_k1.imag,label='$\\tau = %.2f,\, |\\vec k_{%d}|= %.2f$' %(tau, idx,ks[idx]))
        nequil = int(t_equil/tau)
        ax[0].axvline(nequil,c='y') #plot how many steps thrown out during reblocking procedure
        ax[1].axvline(nequil,c='y') #plot how many steps thrown out during reblocking procedure
    fig.suptitle('$r_s = $' + str(r_s))
    ax[0].legend(loc=1)
    ax[1].legend()
    ax[1].set_xlabel('time step')
    ax[0].set_ylabel('$\Re(f_k)$')
    ax[1].set_ylabel('$\Im(f_k)$')
    plt.tight_layout()
    plt.show()

def Phonon_Mom_Density(df):
    '''
    n_k = a*a = h* f as a function of wave vector k magnitude |k|
    '''    
    
    fig,ax = plt.subplots(2,2,figsize=(6,4.5),sharex='col')
    r_s = df['r_s'].values[0]
    print(r_s)
    taus = np.unique(df['tau'].values)
    fig2,ax2 = plt.subplots()
    for tau in taus:
        df2 = df[df['tau']==tau]
        n_ks = df2['n_ks'].values[-1] #pull out equil config of mom densities
        h_ks = df2['h_ks'].values[0]
        f_ks = df2['f_ks'].values[-1]
        ks = df2['ks'].values[0]
        
        ax[0,0].plot(ks,f_ks.real,'.',label='$f, \\tau = %.2f$' %(tau,) )
        ax[1,0].plot(ks,f_ks.imag,'.',label='$f,\\tau = %.2f$' %(tau,))
        ax[0,0].plot(ks,h_ks.real,'.',label='$h, \\tau = %.2f$' %(tau,) )
        ax[1,0].plot(ks,h_ks.imag,'.',label='$h,\\tau = %.2f$' %(tau,))
        ax[0,1].plot(ks,n_ks.real,'.',label='$\\tau=%.2f$' %(tau,))
        ax[1,1].plot(ks,n_ks.imag,'.',label='$\\tau=%.2f$' %(tau,))
        fmag = np.abs(f_ks)
        hmag = np.abs(h_ks)
        ax2.plot(ks,fmag,label='f')
        ax2.plot(ks,hmag,label='h')
    ax2.legend()
    ax2.set_xlabel('k')
    ax2.set_ylabel('ph amp magnitudes')  
    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()
    ax[1,0].set_xlabel('$|\\vec k|$')
    ax[1,1].set_xlabel('$|\\vec k|$')
    ax[0,0].set_ylabel('$\Re(f_k)$')
    ax[1,0].set_ylabel('$\Im(f_k)$')
    ax[0,1].set_ylabel('$\Re(n_k)$')
    ax[1,1].set_ylabel('$\Im(n_k)$')
    #ax[0].set_ylim(bottom=0)
    #ax[1].set_ylim(bottom=0)
    fig.suptitle('$r_s = %d,\, N_k = %d$' %(r_s,len(ks)))
    plt.tight_layout()
    plt.show()

def Test_DOS_phonon_amp(filenames,scaled=True):
    '''
    Overlay amplitudes of phonons (vs wavevector magnitude k) corresponding to different electron-phonon couplings g, rescaling said amplitudes to be in units of g. Also plot a log-log plot of the amplitudes to look for power law behavior as k-> inf. Semiclassically should be power law at long wavelengths (k->0) and die out faster around the phonon length scale l=sqrt(hw/m).
    Should observe a 1/k power law 
    '''    
    fig,ax = plt.subplots(2,2,figsize=(6,4.5),sharex='col')
    for name in filenames: 
        df = pd.read_pickle(name)
        r_s = df['r_s'].values[0]
        dos = df['g'].values[0]
        taus = np.unique(df['tau'].values)
        for tau in taus:
            df2 = df[df['tau']==tau]
            #plot f_k amps in units of g
            f_ks = df2['f_ks'].values[-1] #get equil. config for phonon amps
            if scaled == True: f_ks = f_ks/dos
            ks = df2['ks'].values[0]
            label = '$g = %.2f$' %(dos,)

            #find and plot log-log of negative f_k values only
            re_idx = f_ks.real != 0
            im_idx = f_ks.imag != 0
            f_abs_re = np.abs(f_ks[re_idx].real)
            f_abs_im = np.abs(f_ks[im_idx].imag)
            logamp_r = np.log10(f_abs_re)
            logamp_i = np.log10(f_abs_im)
            logx = np.log10(ks)
            logx_r = logx[re_idx]
            logx_i = logx[im_idx]
            print(logamp_i)
            ax[0,0].plot(ks,f_ks.real,'.',label=label)
            ax[1,0].plot(ks,f_ks.imag,'.',label=label)
            #only plot points on log log < -5 (isolate negative curve)
            plotidx = logamp_i > -5
            logamp_i = logamp_i[plotidx]
            logx_i = logx_i[plotidx] 
            ax[0,1].plot(logx_r,logamp_r,'.',label=label)
            ax[1,1].plot(logx_i,logamp_i,'.',label=label)
    #interpolate the log-log plot - just need to do the last file since they're all the same
    interpx = np.linspace(logx[0],0.5,100)
    interpy = np.interp(interpx,logx_i,logamp_i)
    #ax[1,1].plot(interpx,interpy,'r')
            
    #log fit to interpolation
    fitimag, txtimag = FitData(logx_i,logamp_i, fit='lin',guess=[1,1],varnames=['\log k','\log(|\Im(f_k)|)'])
    ax[1,1].plot(logx_i,fitimag,label='fit')
    ax[1,1].text(0.3, 0.2, txtimag, horizontalalignment='center', verticalalignment='center', transform=ax[1,1].transAxes)
    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend(loc=1)
    ax[1,0].set_xlabel('$k$')
    ax[1,1].set_xlabel('$\log(k)$')
    ylab = 'f_k' 
    if scaled == True: ylab = 'f_k/g' 
    ax[0,0].set_ylabel('$\Re( %s)$'%(ylab,))
    ax[1,0].set_ylabel('$\Im( %s$' %(ylab,))
    ax[0,1].set_ylabel('$\log(|\Re(%s)|)$' %(ylab,))
    ax[1,1].set_ylabel('$\log(|\Im(%s)|)$'%(ylab,))
    fig.suptitle('$r_s = %d$' %(r_s,))
    plt.tight_layout()
    plt.show()

def PlotReblockedE_vs_L(df):
    '''Plot reblocked energy for different r_s (i.e. system sizes L) as function of L'''
    fig,ax = plt.subplots(1,2,figsize=(7,4.5))
    rs = df['r_s'].values
    Ls = (4*np.pi*2/3)**(1/3) * rs #sys size/length measured in a0; multiply by 2 since 2 = # of electrons
    idxs = np.argsort(Ls)
    Ls = Ls[idxs]
    Es = df['eavg'].values[idxs]
    E_err = df['err'].values[idxs]
    taus = df['tau'].values[idxs]
    ax[0].plot(Ls,Es,'-o')
    ax[0].set_ylabel('$E_{avg}$')
    ax[0].set_xlabel('$L$')
    print(Es)
    print(Ls)
    logx = np.log10(Ls)
    logy = np.log10(-Es)
    print(logx)
    print(logy)
    ax[1].plot(logx,logy,'.')
    ax[1].set_xlabel('$\log L$')
    ax[1].set_ylabel('$\log(-E)$')
    
    fig.suptitle('$\\alpha=%d$'%(df['alpha'].values[0]))
    fit, txt = FitData(logx,logy, yerr=np.log10(E_err), fit='lin',guess=[-1,0],varnames=['\log L','\log(-E)'])
    ax[1].plot(logx,fit,label='fit')
    ax[1].text(0.7, 0.7, txt, horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)
    ax[1].legend() 
    plt.tight_layout()
    plt.show()      

def PlotReblockedE_vs_kcut(df):
    '''Plot reblocked energy for different momentum cutoff values (k_cut = 2pi N/L)'''
    eta = 0.2 #unfortunately hard-coded for now
    l = 20
    alpha = (1-eta)*l
    fig,ax = plt.subplots(1,1,figsize=(6,4.5))
    rs = df['r_s'].values
    Ncuts = df['Ncut'].values
    Ls = (4*np.pi*2/3)**(1/3) * rs #sys size/length measured in a0; multiply by 2 since 2 = # of electrons
    idxs = np.argsort(Ls)
    Ls = Ls[idxs]
    kcuts = 2*np.pi*Ncuts[idxs]/Ls
    Es = df['eavg'].values[idxs]
    E_err = df['err'].values[idxs]
    ax.plot(kcuts,Es,'-o',label='data')
    ax.set_ylabel('$E_{avg}$')
    ax.set_xlabel('$k_{cut}=2\pi N/L$')
    dist = Ls*np.sqrt(3)/2 #inter-electron distance; for bcc setup (one e- at center, one at far corner) d = L sqrt(3)/2
    anafit,t_ana = FitData(kcuts,Es,varnames=['k_{cut}','E'])
    print(t_ana)
    slope = -2/np.pi*(1-eta)
    #slope = 0.1
    pred = slope*kcuts + eta/dist #semiclassical prediction E_bi(r) = -(1-eta)*2*k_cut/pi + eta/r
    pred_llp = -alpha/(2*l**2)*2
    ax.plot(kcuts,pred,'ko-',label='semiclassical')
    ax.axhline(pred_llp,c='red',label='-2alpha hw')
    ax.set_title('$\\alpha= %d$, $r_s = %d$' %(alpha,rs[0]))
    ax.legend() 
    plt.tight_layout()
    plt.show()      

def PlotVars(df, xvar=['step'], yvars=['elocal']):
    '''
    Plot variables from the direct DMC results. If want to plot multiple y variables, input should be as an array
    '''
    taus = np.unique(df['tau'].values)
    fig,ax = plt.subplots(2,1,figsize=(6,4.5),sharex='row')
    r_s = df['r_s'].values[0]
    print(r_s)
    for tau in taus:
        df2 = df[df['tau']==tau]
        steps = df2[xvar].values
        for name in yvars:
            Earray = df2[name].values
            eloc = np.array([complex(val) for val in Earray]) 
            ax[0].plot(steps,eloc.real,label='Re(' + name + ')')
            ax[1].plot(steps,eloc.imag,label='Im(' + name + ')')
        
        plt.title('$r_s = $' + str(r_s))
        nequil = int(t_equil/tau)
        ax[0].axvline(nequil,c='y') #plot how many steps thrown out during reblocking procedure
        ax[1].axvline(nequil,c='y') #plot how many steps thrown out during reblocking procedure
        ax[0].legend()
        ax[1].legend()
        ax[1].set_xlabel('time step')
        ax[0].set_ylabel('Re(E)')
        ax[1].set_ylabel('Im(E)')
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

def FitData(xvals, yvals, yerr=[], fit='lin', extrap=[],guess=[-1,-3],varnames=['\\tau','\log E'],retparam=False):
    def fitlinear(x,a,b):
        f = a*x + b 
        return f

    bnds = ([-30,-30],[10,10]) #bounds for weak coupling fit
    xname, yname = varnames
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
        r'$%s = a%s + b$' %(yname,xname),
        r'$a=%.4f \pm %.4f$' % (a, aerr),
        r'$b=%.4f \pm %.4f$' % (b, berr)
        ))

    print(r'$b=%.5f \pm %.5f$' % (b, berr))
    if retparam == True:
        return param, ans, textstr
    else:
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
    #df = pd.read_pickle(sys.argv[1])
    #PlotErr(df,yvar='eavg',err='err')
    #PlotVars(df,yvars=['elocal','egth'])
    #Emix_vs_gth(df)
    #Timeplot_phonon_amps(df)
    #Phonon_Mom_Density(df)
    #RandTrials(sys.argv[1:])

    df = pd.read_csv(sys.argv[1])
    PlotReblockedE_vs_kcut(df)
    #PlotReblockedE_vs_L(df)
 
    #filenames = sys.argv[1:]
    #Test_DOS_phonon_amp(filenames,scaled=False)
    #df = pd.read_csv(sys.argv[1])
    #Test3(df) #need to use reblocked energy file
    #Extrapolate_Emix_gth(df)
