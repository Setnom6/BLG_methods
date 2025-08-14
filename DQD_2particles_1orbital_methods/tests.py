from src.DQD_2particles_1orbital import DQD_2particles_1orbital, DQDParameters
from src.LindblandOperator import LindbladOperator

from qutip import *
import pymablock
import numpy as np
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt

gOrtho = 10
U0 = 8.5
U1 = 0.1
Ei = 8.25
bx = 0.179
fixedParameters = {
            DQDParameters.B_FIELD.value: 0.20,
            DQDParameters.B_PARALLEL.value: bx,
            DQDParameters.E_I.value: Ei,
            DQDParameters.T.value: 0.004,
            DQDParameters.DELTA_SO.value: 0.06,
            DQDParameters.DELTA_KK.value: 0.02,
            DQDParameters.T_SOC.value: 0.0,
            DQDParameters.U0.value: U0,
            DQDParameters.U1.value: U1,
            DQDParameters.X.value: 0.02,
            DQDParameters.G_ORTHO.value: gOrtho,
            DQDParameters.G_ZZ.value: 10 * gOrtho,
            DQDParameters.G_Z0.value: 2 * gOrtho / 3,
            DQDParameters.G_0Z.value: 2 * gOrtho / 3,
            DQDParameters.GS.value: 2,
            DQDParameters.GSLFACTOR.value: 1.0,
            DQDParameters.GV.value: 20.0,
            DQDParameters.GVLFACTOR.value: 0.66,
            DQDParameters.A.value: 0.1,
            DQDParameters.P.value: 0.02,
            DQDParameters.J.value: 0.00075 / gOrtho,
}

dqd = DQD_2particles_1orbital(fixedParameters)


N0 = 5
N1 = 6
H = dqd.project_hamiltonian(dqd.singlet_triplet_reordered_basis)
LM = LindbladOperator(dqd.FSU)
Li = LM.buildDephasingOperator(3)
Li_proj = dqd.project_hamiltonian(dqd.singlet_triplet_reordered_basis, alternative_operator=Li)[:N0+N1, :N0+N1]


print(Li_proj[:5,:5])

subspace_indices = [0]*N0 + [1]*N1
H0_tot = H[:N0+N1, :N0+N1]
H0 = np.diag(np.diag(H0_tot))
H1 = H0_tot-H0

hamiltonian = [H0, H1]

H_tilde, U_bs, U_adj_bs = pymablock.block_diagonalize(hamiltonian, subspace_indices=subspace_indices)
H_eff =  np.ma.sum(H_tilde[:2, :2, :3], axis=2)

Li_proj_series = pymablock.operator_to_BlockSeries(
    [np.diag(np.diag(Li_proj)), Li_proj-np.diag(np.diag(Li_proj))],
    name="H0_tot",
    hermitian=True, 
    subspace_indices=subspace_indices
)

Li_proj_eff = pymablock.series.cauchy_dot_product(U_adj_bs, Li_proj_series, U_bs)
Li_eff = np.ma.sum(Li_proj_eff[:2, :2, :3], axis=2)

print(Li_eff[0,0])

        

