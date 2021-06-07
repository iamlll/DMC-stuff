import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def PlotE(csvname):
    df = pd.read_csv(csvname)

    fig = plt.figure(figsize=(10,4.5))
    ax = fig.add_subplot(111)
    #read in CSV as Pandas dataframe
    xs = df['step'].values.real
    wts = df['weight'].values.real
    elocs = df['elocal'].values.real
    elocvars = df['elocalvar'].values.real
    erefs = df['eref'].values.real
    print(len(elocs))
    elocs2 = [elocs[i] for i in range(0,len(elocs), 10)] #select every 10th element
    print(len(elocs2))
    ax.plot(xs, elocs,label='$E_{loc}$')
    #ax.plot(xs, wts,label='$weights$')
    #ax.plot(xs, erefs,label='$E_{ref}$')
    #ax.plot(xs, elocvars,label='$\sigma_{loc}$')
    ax.set_xlabel("step")
    #ax.set_ylabel("$\Delta E/|E_\infty|$")
    #ax.set_ylim(-0.2,0.2)

    ax.legend(loc=1)

    plt.tight_layout()
    plt.show()

PlotE("phonon_mc.csv")
