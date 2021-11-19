#!/usr/bin/env python3
import os
import h5py
import numpy as np
from pyscf.pbc import gto as pbcgto
import pyqmc
from pyqmc.coord import PeriodicConfigs 
from egas import EnergyAccumulator
from pyqmc.wftools import generate_jastrow
from pyqmc.dmc import rundmc

def main():
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('--rs', type=float, default=4)
  parser.add_argument('--dt', type=float)
  parser.add_argument('--nconf', type=int, default=512)
  parser.add_argument('--fopt', type=str)
  parser.add_argument('--nblock_ref', type=int, default=1000)
  parser.add_argument('--verbose', action='store_true')
  args = parser.parse_args()

  rs = args.rs  # inter-electron spacing, controls density
  nelec = 2
  ndim = 3
  lbox = (4*np.pi/3*nelec)**(1/3) * rs  # sys. size/length measured in a0; multiply by 2 since 2 = # of electrons

  axes = lbox*np.eye(ndim)
  pos = lbox*np.random.rand(args.nconf, nelec, ndim)

  # design a systemmatic prefix for output file
  tref = rs/10
  tstep = args.dt
  if tstep is None:
    tstep = tref
  nmult = int(max(1, tref//tstep))
  nconf = args.nconf
  prefix = 'rs%d-dt%.2f-n%d' % (rs, tstep, nconf)

  fopt = args.fopt
  if fopt is None:
    fopt = 'opt-rs%d.h5' % rs
  if not os.path.isfile(fopt):
    msg = 'need file with optimized Jastrow parameters\n'
    msg += '  %s not found' % fopt
    raise RuntimeError(msg)
  # read optimzed Jastrow parameters
  with h5py.File(fopt, 'r') as f:
    bcoeff = f['wf']['bcoeff'][()]

  nblocks = args.nblock_ref*nmult

  # simulation cell
  cell = pbcgto.M(
    atom = 'He 0 0 0',
    a = axes,
    unit='B',  # B = units of Bohr radii
  )
  # ee Jastrow (only bcoeff for ee, no acoeff for ei)
  wf, to_opt = generate_jastrow(cell, ion_cusp=[], na=0)
  wf.parameters['bcoeff'] = bcoeff  # given by step1_opt.py
  # initialize electrons uniformly inside the box
  configs = PeriodicConfigs(pos, axes)
  # use hacked energy in estimator
  accus = {"energy": EnergyAccumulator(cell)}
  data, confs, wts = rundmc(
    wf,
    configs,
    accumulators=accus,
    hdf_file='%s-dmc.h5' % prefix,
    nsteps=nblocks,
    tstep=tstep,
    verbose=True,
  )

  # quick analysis (guess tequil and tcorr)
  trace = data['energytotal']
  tequil = 10
  tcorr = 1
  nskip = int(round(tequil/tstep))
  nevery = int(round(tcorr/tstep))
  trace1 = trace[nskip::nevery]
  ym = np.mean(trace1)
  ye = np.std(trace1, ddof=1)
  print(ym)
  print(ye)
  

if __name__ == '__main__':
  main()
