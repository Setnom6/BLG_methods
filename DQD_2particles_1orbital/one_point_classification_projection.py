from src.DQD_2particles_1orbital import DQD_2particles_1orbital, DQDParameters
import numpy as np
from scipy.linalg import eigh
gOrtho = 1
U0 = 8.5
# Example usage
fixedParameters = {
        DQDParameters.B_FIELD.value: 0.5,  # Set B-field
        DQDParameters.B_PARALLEL.value: 0.0,  # Set parallel magnetic field to zero
        DQDParameters.E_I.value: 2.0,  # Set detuning to zero
        DQDParameters.T.value: 0.04,  # Set hopping parameter
        DQDParameters.DELTA_SO.value: 0.06,  # Set Kane
        DQDParameters.DELTA_KK.value: 0.02,  # Set valley mixing
        DQDParameters.T_SOC.value: 0.01,  # Set spin-flip
        DQDParameters.U0.value: U0,  # Set on-site Coul
        DQDParameters.U1.value: 0.1,  # Set nearest-neighbour Coulomb potential
        DQDParameters.X.value: 0.02,  # Set intersite exchange interaction
        DQDParameters.G_ORTHO.value: gOrtho,  # Set orthogonal component of tunneling corrections
        DQDParameters.G_ZZ.value: 10*gOrtho,  # Set correction along
        DQDParameters.G_Z0.value: -2*gOrtho/3,  # Set correction for Z-O mixing
        DQDParameters.G_0Z.value: -2*gOrtho/3,  # Set correction for O-Z mixing
        DQDParameters.GS.value: 2.0,  # Set spin g-factor
        DQDParameters.GV.value: 20.0,  # Set valley g-factor
        DQDParameters.A.value: 0.1,  # Set density assisted hopping
        DQDParameters.P.value: 0.02,  # Set pair hopping interaction
        DQDParameters.J.value: 0.075/gOrtho,  # Set renormalization parameter for Coulomb corrections
}

number_of_eigenstates = 16  # Number of eigenstates to plot


dqd = DQD_2particles_1orbital()
basis_to_project = dqd.original_basis
correspondence = dqd.original_correspondence
lenBasis = len(basis_to_project)

parameters_to_change = fixedParameters.copy() if fixedParameters is not None else {}
parameters_to_change[DQDParameters.B_FIELD.value] = 0.1  # Set a small B-field to avoid degeneracy
projectedH = dqd.project_hamiltonian(basis_to_project, parameters_to_change=parameters_to_change)

dqd.diagnoseProjectionQuality(basis_to_project, parameters_to_change)
print("-----------------------------\n")
eigval, eigv = eigh(projectedH)
    

preferred_basis = []
for i in range(lenBasis):
    preferred_basis.append(np.array([0]*i+[1]*1+[0]*(lenBasis-i-1)))

for i in range(number_of_eigenstates):
    classification = dqd.FSU.classify_eigenstate(preferred_basis,correspondence,eigv[:, i])
    print(f"Eigenstate {i+1}:")
    print(f"Eigenvalue: {eigval[i]:.4f} meV")
    print(f"Most similar state (Dot, Spin, Valley): {classification['most_similar_state']}")
    print(f"Probability: {classification['probability']:.4f}")
    for i in range(1,4):
        print(f"{i+1} order in similarity: {classification['ordered_probabilities'][i]['label']}")
        print(f"with probability: {classification['ordered_probabilities'][i]['probability']}")
    print("\n")