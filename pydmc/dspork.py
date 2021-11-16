#!/usr/bin/env python3
import h5py
import numpy as np
import pandas as pd

def read_traces(fh5):
  nblock = None
  fp = h5py.File(fh5, 'r')
  data = {}
  for key in fp.keys():
    grp = fp[key]
    if len(grp.shape) == 1:
      if nblock is None:
        nblock = grp.shape[0]
      else:
        if nblock != grp.shape[0]:
          continue
      data[key] = grp[()]
  fp.close()
  df = pd.DataFrame(data)
  return df

def main():
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('fh5', type=str)
  parser.add_argument('--nequil', '-e', type=int, default=0)
  parser.add_argument('--column', '-c', type=str, default='energytotal')
  parser.add_argument('--ndigit', '-d', type=int, default=6)
  parser.add_argument('--plot', '-t', action='store_true')
  parser.add_argument('--list_columns', '-l', action='store_true')
  parser.add_argument('--verbose', action='store_true')
  args = parser.parse_args()

  df = read_traces(args.fh5)
  if args.list_columns:
    print(df.columns)
  #trace = df[args.column].values
  sel = df.index > args.nequil
  ym = None
  try:
    from qharv.reel.scalar_dat import single_column
    ym, ye, yc = single_column(df.loc[sel], args.column, 0)
    print(np.round(ym, args.ndigit), np.round(ye, args.ndigit), np.round(yc, 2))
  except ImportError:
    msg = 'failed to import qharv, please download harvest_qmcpack GitHub repo or do your own statistics'
    print(msg)
  if args.plot:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    if ym is not None:
      ax.axhline(ym, c='k')
    ax.axvline(args.nequil, c='b')
    ax.plot(df[args.column])
    plt.show()

if __name__ == '__main__':
  main()  # set no global variable
