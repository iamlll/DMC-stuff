import numpy as np
import pyqmc.energy as energy
import pyqmc.ewald as ewald
#import pyqmc.eval_ecp as eval_ecp
# copy from pyqmc.accumulators
#  zero out ei, ii, ecp_val

class EnergyAccumulator:
    """Returns local energy of each configuration in a dictionary."""

    def __init__(self, mol, threshold=10, **kwargs):
        self.mol = mol
        self.threshold = threshold
        if hasattr(mol, "a"):
            self.coulomb = ewald.Ewald(mol, **kwargs)
        else:
            self.coulomb = energy.OpenCoulomb(mol, **kwargs)

    def __call__(self, configs, wf):
        ee, ei, ii = self.coulomb.energy(configs)
        ei.fill(0)
        ii = 0
        #ecp_val = eval_ecp.ecp(self.mol, configs, wf, self.threshold)
        ecp_val = np.zeros(len(configs.configs))
        ke, grad2 = energy.kinetic(configs, wf)
        ke = ke
        return {
            "ke": ke,
            "ee": ee,
            "ei": ei,
            "ecp": ecp_val,
            "grad2": grad2,
            "total": ke + ee + ei + ecp_val + ii,
        }

    def avg(self, configs, wf):
        return {k: np.mean(it, axis=0) for k, it in self(configs, wf).items()}

    def nonlocal_tmoves(self, configs, wf, e, tau):
        return eval_ecp.compute_tmoves(self.mol, configs, wf, e, self.threshold, tau)

    def has_nonlocal_moves(self):
        return self.mol._ecp != {}

    def keys(self):
        return set(["ke", "ee", "ei", "ecp", "total", "grad2"])

    def shapes(self):
        return {"ke": (), "ee": (), "ei": (), "ecp": (), "total": (), "grad2": ()}

