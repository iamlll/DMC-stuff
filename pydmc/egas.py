#!/usr/bin/env python3
import numpy as np
from pyqmc.ewald import Ewald
from pyqmc.energy import kinetic
#copied from pyqmc class, set EI & II = 0
class EnergyAccumulator:
    def __init__(self, mol, threshold=10, **kwargs):
        self.mol = mol
        self.threshold = threshold
        if hasattr(mol, "a"):
            self.ewald = Ewald(mol, **kwargs)
            def compute_energy(mol, configs, wf, threshold):
                ee, ei, ii = self.ewald.energy(configs)
                ei = 0
                ii = 0
                ke = kinetic(configs, wf)
                return {
                    "ke": ke,
                    "ee": ee,
                    "total": ke + ee,
                }

            self.compute_energy = compute_energy
        else:
            self.compute_energy = energy.energy

    def __call__(self, configs, wf):
        return self.compute_energy(self.mol, configs, wf, self.threshold)

    def avg(self, configs, wf):
        d = {}
        for k, it in self(configs, wf).items():
            d[k] = np.mean(it, axis=0)
        return d

    def keys(self):
        return set(["ke", "ee", "ei", "total"])

    def shapes(self):
        return {"ke": (), "ee": (), "ei": (), "total": ()}

