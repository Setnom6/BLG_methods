from pymablock import block_diagonalization
from src.DQD_2particles_1orbital import DQD_2particles_1orbital, DQDParameters
from plotting_methods import obtain_dict_parameters_to_change, obtain_dict_labels, plot_hamiltonian_no_blocks, plot_hamiltonian_charge_configuration_blocks, BasisToProject
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, block_diag, orth

gOrtho = 1
U0 = 8.5
# Example usage
fixedParameters = {
        DQDParameters.B_FIELD.value: 0.75,  # Set B-field
        DQDParameters.B_PARALLEL.value: 0.0,  # Set parallel magnetic field to zero
        DQDParameters.E_I.value: 0.0,  # Set detuning to zero
        DQDParameters.T.value: 0.04,  # Set hopping parameter
        DQDParameters.DELTA_SO.value: 0.06,  # Set Kane
        DQDParameters.DELTA_KK.value: 0.02,  # Set valley mixing
        DQDParameters.T_SOC.value: 0.01,  # Set spin-flip
        DQDParameters.U0.value: U0,  # Set on-site Coul
        DQDParameters.U1.value: 0.1,  # Set nearest-neighbour Coulomb potential
        DQDParameters.X.value: 0.02,  # Set intersite exchange interaction
        DQDParameters.G_ORTHO.value: gOrtho,  # Set orthogonal component of tunneling corrections
        DQDParameters.G_ZZ.value: 0.1*gOrtho,  # Set correction along
        DQDParameters.G_Z0.value: 2*gOrtho/3,  # Set correction for Z-O mixing
        DQDParameters.G_0Z.value: 2*gOrtho/3,  # Set correction for O-Z mixing
        DQDParameters.GS.value: 2.0,  # Set spin g-factor
        DQDParameters.GV.value: 20.0,  # Set valley g-factor
        DQDParameters.A.value: 0.1,  # Set density assisted hopping
        DQDParameters.P.value: 0.02,  # Set pair hopping interaction
        DQDParameters.J.value: 0.075/gOrtho,  # Set renormalization parameter for Coulomb corrections
}

basis = BasisToProject.ORIGINAL

dqd = DQD_2particles_1orbital()
dict_relations = {
        BasisToProject.SINGLET_TRIPLET_BASIS: (dqd.singlet_triplet_basis, dqd.singlet_triplet_correspondence, None),
        BasisToProject.SPIN_SYMMETRY: (dqd.spin_symmetry_basis, dqd.symmetric_antisymmetric_correspondence, {"Sym": 6, "AntiSym": 10, "ortho": 12}),
        BasisToProject.VALLEY_SYMMETRY: (dqd.valley_symmetry_basis, dqd.symmetric_antisymmetric_correspondence, {"Sym": 6, "AntiSym": 10, "ortho": 12}),
        BasisToProject.ORBITAL_SYMMETRY: (dqd.orbital_symmetry_basis, dqd.symmetric_antisymmetric_correspondence, {"Sym": 6, "AntiSym": 10, "ortho": 12}),
        BasisToProject.ORIGINAL: (dqd.original_basis, dqd.original_correspondence, None)
}


basis_to_project = dict_relations[basis][0]
correspondence = dict_relations[basis][1]
name_basis = basis.value
blocks_dict = dict_relations[basis][2]

parameters_to_change = fixedParameters.copy() if fixedParameters is not None else {}
parameters_to_change[DQDParameters.B_FIELD.value] = 0.1  # Set a small B-field to avoid degeneracy

H_full = dqd.obtain_hamiltonian_determinant_basis(parameters_to_change=parameters_to_change)
dimFull = H_full.shape[0]
I  = np.eye(dimFull)
U_P = np.array(basis_to_project).T

if U_P.shape[1] < 28:
        P_proj = U_P @ U_P.conj().T  # (28,28)
        Q_proj = I - P_proj
        U_Q = orth(Q_proj)

        extra_basis = [vector for vector in U_Q.T]

        total_basis_to_project = basis_to_project + extra_basis

        H_PP = U_P.conj().T @ H_full @ U_P
        H_PQ = U_P.conj().T @ H_full @ U_Q
        H_QP = H_PQ.conj().T
        H_QQ = U_Q.conj().T @ H_full @ U_Q

else:
        total_basis_to_project = basis_to_project
        

HProjectedTotal = dqd.project_hamiltonian(total_basis_to_project, parameters_to_change=parameters_to_change)
dqd.diagnoseProjectionQuality(total_basis_to_project, parameters_to_change)
print("-----------------------------\n")


plot_hamiltonian_charge_configuration_blocks(np.abs(HProjectedTotal), title= f"Hamiltonian in {name_basis}", alternative_blocks=blocks_dict)
plt.show()