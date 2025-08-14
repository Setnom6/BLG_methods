import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.DQD_2particles_1orbital import DQD_2particles_1orbital, DQDParameters
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import inv, eigh
import os
from datetime import datetime
from pymablock import block_diagonalize

def schriefferWolff(H_full):
    N0 = 5
    N1 = 6
    subspace_indices = [0]*N0 + [1]*N1
    H0_tot = H_full[:N0+N1, :N0+N1]
    H0 = np.diag(np.diag(H0_tot))
    H1 = H0_tot-H0

    hamiltonian = [H0, H1]

    H_tilde, _, _ = block_diagonalize(hamiltonian, subspace_indices=subspace_indices)

    transformed_H = np.ma.sum(H_tilde[:2, :2, :3], axis=2)
    return transformed_H[0, 0]

def firstOrderNaif(H_full):
    N = 11
    H00 = H_full[:N, :N]
    H01 = H_full[:N, N:]
    H10 = H_full[N:, :N]
    H11 = H_full[N:, N:]
    return H00

# Fixed Parameters
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

# Initialize
dqd = DQD_2particles_1orbital(fixedParameters)

# Move detuning Ei
eiValues = np.linspace(0.75 * U0, 1.2 * U0, 1000)
allEigenvalues_full = []
allEigenvalues_SWT = []
allEigenvalues_naif = []
allEigenvalues_H00 = []

basis_to_project = dqd.singlet_triplet_reordered_basis
N = 5

for ei in eiValues:
    parameters = fixedParameters.copy()
    parameters[DQDParameters.E_I.value] = ei

    H_full = dqd.project_hamiltonian(basis_to_project, parameters_to_change=parameters)

    # Decouple states of interest
    H00 = H_full[:N, :N]

    H_eff_SWT = schriefferWolff(H_full)
    H_eff_naif = firstOrderNaif(H_full)

    # Diagonalizations
    eigenvalues_full, _ = eigh(H_full)
    eigenvalues_SWT, _ = eigh(H_eff_SWT)
    eigenvalues_naif, _ = eigh(H_eff_naif)
    eigenvalues_H00, _ = eigh(H00)

    allEigenvalues_full.append(eigenvalues_full[:N])
    allEigenvalues_SWT.append(eigenvalues_SWT)
    allEigenvalues_naif.append(eigenvalues_naif)
    allEigenvalues_H00.append(eigenvalues_H00)

# Convert to numpy arrays
allEigenvalues_full = np.array(allEigenvalues_full)
allEigenvalues_SWT = np.array(allEigenvalues_SWT)
allEigenvalues_naif = np.array(allEigenvalues_naif)
allEigenvalues_H00 = np.array(allEigenvalues_H00)

# Compute absolute errors
errors_SWT = np.abs(allEigenvalues_full[:, :N] - allEigenvalues_SWT[:, :N])
errors_naif = np.abs(allEigenvalues_full[:, :N] - allEigenvalues_naif[:, :N])
errors_H00 = np.abs(allEigenvalues_full[:, :N] - allEigenvalues_H00[:, :N])

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# 1. First order vs Full
for i in range(N):
    axs[0].plot(eiValues, allEigenvalues_full[:, i] - eiValues, 'C' + str(i), label=f'Full {i+1}')
    axs[0].plot(eiValues, allEigenvalues_SWT[:, i] - eiValues, 'C' + str(i) + '--', label=f'SW {i+1}')
    #axs[0].plot(eiValues, allEigenvalues_naif[:, i] - eiValues, 'C' + str(i) + '.', label=f'FoN {i+1}')
    axs[0].plot(eiValues, allEigenvalues_H00[:, i] - eiValues, 'C' + str(i) + ':', label=f'H00 {i+1}')
axs[0].set_ylabel("Energy - $E_I$ (meV)")
axs[0].set_title("First order and H00 vs Full Hamiltonian")
axs[0].legend()

# 3. Error First order vs Full
for i in range(N):
    axs[1].plot(eiValues, errors_SWT[:, i], 'C' + str(i) + '--', label=f'Level {i} SWT')
    #axs[1].plot(eiValues, errors_naif[:, i], 'C' + str(i) + '.', label=f'Level {i} FoN')
    axs[1].plot(eiValues, errors_H00[:, i], 'C' + str(i)  + ':', label=f'Level {i} H00')
axs[1].set_yscale("log")
axs[1].set_xlabel("$E_I$ (meV)")
axs[1].set_ylabel("Abs. Error (meV)")
axs[1].set_title("Error |SWT - Full| and |H00 - Full|")
axs[1].legend()

plt.tight_layout()
figures_dir = os.path.join(os.getcwd(),"DQD_2particles_1orbital_methods", "figures")
os.makedirs(figures_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

fig_path = os.path.join(figures_dir, f"First order_{timestamp}.png")
plt.savefig(fig_path)
print(f"Figure saved in: {fig_path}")

param_path = os.path.join(figures_dir, f"parameters_First_order_approx_{timestamp}.txt")
with open(param_path, 'w') as f:
    for key, value in fixedParameters.items():
        f.write(f"{key}: {value}\n")
print(f"Parameters saved in: {param_path}")
plt.show()


