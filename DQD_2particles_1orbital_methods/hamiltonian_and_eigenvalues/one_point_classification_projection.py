import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.DQD_2particles_1orbital import DQD_2particles_1orbital, DQDParameters
import numpy as np
from scipy.linalg import eigh
gOrtho = 10
U0 = 8.5
U1 = 0.1
Ei = 0.0
fixedParameters = {
    DQDParameters.B_FIELD.value: 0.2,
    DQDParameters.B_PARALLEL.value: 0.15,
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

number_of_eigenstates = 5  # Number of eigenstates to plot


dqd = DQD_2particles_1orbital()
basis_to_project = dqd.singlet_triplet_reordered_basis
correspondence = dqd.singlet_triplet_reordered_correspondence
lenBasis = len(basis_to_project)

projectedH = dqd.project_hamiltonian(basis_to_project, parameters_to_change=fixedParameters)

dqd.diagnoseProjectionQuality(basis_to_project, fixedParameters)
print("-----------------------------\n")
eigval, eigv = eigh(projectedH)
    

preferred_basis = []
for i in range(lenBasis):
    preferred_basis.append(np.array([0]*i+[1]*1+[0]*(lenBasis-i-1)))

for i in range(number_of_eigenstates):
    classification = dqd.FSU.classify_eigenstate(preferred_basis,correspondence,eigv[:, i])
    print(f"Eigenstate {i+1}:")
    print(f"Eigenvalue (-E_i): {eigval[i] - fixedParameters[DQDParameters.E_I.value]:.4f} meV")
    print(f"Most similar state (Dot, Spin, Valley): {classification['most_similar_state']}")
    print(f"Probability: {classification['probability']:.4f}")
    for i in range(1,4):
        print(f"{i+1} order in similarity: {classification['ordered_probabilities'][i]['label']}")
        print(f"with probability: {classification['ordered_probabilities'][i]['probability']}")
    print("\n")