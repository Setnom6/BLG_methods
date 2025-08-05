from src.DQD_2particles_1orbital import DQD_2particles_1orbital, DQDParameters
import matplotlib.pyplot as plt
from scipy.linalg import inv
from qutip import *
import numpy as np
import os
from datetime import datetime
import multiprocessing
from copy import deepcopy
from functools import partial




def runDynamics(fixedParameters, bx, tTotal=2.5):

    # Sweep protocol reduced to just the rabi oscillations
    totalPoints = 300
    nsToMeV = 1519.29

    dqd = DQD_2particles_1orbital(fixedParameters)
    basis = dqd.singlet_triplet_reordered_basis
    correspondence = dqd.singlet_triplet_reordered_correspondence
    inverseCorrespondence = {v: k for k, v in correspondence.items()}
    N = 11  # Size of H00 block

    # === Time vectors (in ns and meV⁻¹) ===
    tlistNano = np.linspace(0,tTotal, totalPoints)
    tlist = nsToMeV * tlistNano

    # Initial state: ground state for 0 detuning
    # Compute H_eff at t = 0 and get its ground state
    params_initial = deepcopy(fixedParameters)
    params_initial[DQDParameters.B_PARALLEL.value] = bx
    params_initial[DQDParameters.E_I.value] = 0.0

    H_full_initial = dqd.project_hamiltonian(basis, parameters_to_change=params_initial)
    H00_initial = H_full_initial[:N, :N]
    H01_initial = H_full_initial[:N, N:]
    H10_initial = H_full_initial[N:, :N]
    H11_initial = H_full_initial[N:, N:]

    hEff_initial = H00_initial - H01_initial @ inv(H11_initial) @ H10_initial
    hEffQobj_initial = Qobj(hEff_initial)
    _, evecs_initial = hEffQobj_initial.eigenstates()
    psi0 = evecs_initial[0]  # ground state
    rho0 = psi0 * psi0.dag()

    # === Effective Hamiltonian with specified detuning ===
    params = deepcopy(fixedParameters)
    params[DQDParameters.B_PARALLEL.value] = bx

    H_full = dqd.project_hamiltonian(basis, parameters_to_change=params)
    H00 = H_full[:N, :N]
    H01 = H_full[:N, N:]
    H10 = H_full[N:, :N]
    H11 = H_full[N:, N:]

    hEff = H00 - H01 @ inv(H11) @ H10
    hEffQobj = Qobj(hEff)

    # === Time evolution ===
    result = mesolve(hEffQobj, rho0, tlist, c_ops=[])
    I_t = []

    for state in result.states:
        population = state.diag()

        totalPopulation = (
            population[inverseCorrespondence["LL,S,T-"]]
            + population[inverseCorrespondence["LR,S,T-"]]
            + population[inverseCorrespondence["LR,T+,T-"]]
            + population[inverseCorrespondence["LR,T0,T-"]]
        )

        I = (
            population[inverseCorrespondence["LL,S,T-"]]
            + population[inverseCorrespondence["LR,S,T-"]]
        ) / totalPopulation

        I_t.append(I)

    return np.array(I_t)



def getCurrentForMagneticField(bx, fixedParameters, tTotal):
    return runDynamics(fixedParameters, bx, tTotal)

def plotCurrentMap(fixedParameters, bxList, tTotal):
    maxCores = 24
    availableCores = multiprocessing.cpu_count()
    numCores = min(maxCores, availableCores)

    with multiprocessing.Pool(processes=numCores) as pool:
        getCurrent = partial(getCurrentForMagneticField, fixedParameters=fixedParameters, tTotal=tTotal)
        results = pool.map(getCurrent, bxList)

    currentMatrix = np.array(results)

    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        currentMatrix,
        aspect="auto",
        origin="lower",
        extent=[0, tTotal, bxList[0], bxList[-1]],
        cmap="viridis"
    )
    plt.colorbar(im, label="I (no Pauli Blockade)")
    plt.xlabel("Time(ns)")
    plt.ylabel("bx (T)")
    plt.title("Current vs bx and interaction time")

    figuresDir = os.path.join(os.getcwd(), "DQD_2particles_1orbital", "figures")
    os.makedirs(figuresDir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    figPath = os.path.join(figuresDir, f"Current_bx_time_{timestamp}.png")
    plt.savefig(figPath)
    print(f"Figure saved at: {figPath}")

    paramPath = os.path.join(figuresDir, f"parameters_Current_bx_time_{timestamp}.txt")
    with open(paramPath, 'w') as f:
        for key, value in fixedParameters.items():
            f.write(f"{key}: {value}\n")
    print(f"Parameters saved at: {paramPath}")


if __name__ == "__main__":
    # Execution

    gOrtho = 10
    U0 = 8.5
    U1 = 0.1
    fixedParameters = {
        DQDParameters.B_FIELD.value: 0.20,
        DQDParameters.B_PARALLEL.value: 0.0,
        DQDParameters.E_I.value: 8.2,
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

    totalTime = 2.5 # ns
    bxValues = np.linspace(0.0, 0.75, 30)  # meV

    plotCurrentMap(fixedParameters, bxValues, totalTime)