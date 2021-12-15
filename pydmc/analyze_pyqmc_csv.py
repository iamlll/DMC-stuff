#!/usr/bin/env python3
import numpy as np
import pandas as pd
from qharv.reel import scalar_dat

def main():
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('fcsv', type=str)
  parser.add_argument('--yname', type=str, default="elocal",
    help="elocal, ke, weight, elocalvar")
  parser.add_argument('--tequil', '-e', type=float, default=10,
    help="equilibration time in 1/ha, default 10")
  parser.add_argument('--tcorr', '-c', type=float, default=0.2,
    help="correlation time in 1/ha, default 0.2")
  parser.add_argument('--plot_trace', '-t', action='store_true')
  parser.add_argument('--verbose', '-v', action='store_true')
  args = parser.parse_args()
  yname = args.yname
  tequil = args.tequil
  tcorr = args.tcorr
  xname = 'step'

  df = pd.read_csv(args.fcsv)

  header = "#    tau      %s_mean     %s_error" % (yname, yname)
  print(header)
  if args.plot_trace:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    ax.set_ylabel(yname)
  for tau in df.tau.unique():
    sel = df.tau == tau
    nequil = int(round(tequil/tau))
    ncorr = max(1, int(round(tcorr/tau)))
    if args.verbose:
      print(tau, nequil, ncorr)
    esel = df[xname] > nequil
    # cut equilibration & autocorrelation
    y = df.loc[sel&esel, yname].values[::ncorr]
    if args.plot_trace:
      ax.plot(y, label=tau)
    # calculate mean and error
    ym = y.mean()
    ye = y.std(ddof=1)/len(y)**0.5
    # output
    line = '%8.4f %16.6f %16.6f' % (tau, ym, ye)
    print(line)
  if args.plot_trace:
    ax.legend()
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
  main()  # set no global variable
