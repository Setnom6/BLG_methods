from src.DQD_2particles_1orbital import DQD_2particles_1orbital, DQDParameters
from scipy.linalg import inv
from qutip import *
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import multiprocessing
from joblib import delayed, Parallel

def simulateCurrentAtTWait(fixedParameters, initialStateLabel, tWait, eiMiddle, totalPoints=500):
    nsToMeV = 1519.29
    dqd = DQD_2particles_1orbital(fixedParameters)
    basis = dqd.singlet_triplet_reordered_basis
    correspondence = dqd.singlet_triplet_reordered_correspondence
    inverseCorrespondence = {v: k for k, v in correspondence.items()}
    N = 5

    tRamp, tJump, tWait2 = 0.6, 0.6, 0.5  # ns
    tTotal = tRamp + tWait + tJump + tWait2

    nRamp = int(tRamp * totalPoints / tTotal)
    nWait = int(tWait * totalPoints / tTotal)
    nJump = int(tJump * totalPoints / tTotal)
    nWait2 = totalPoints - (nRamp + nWait + nJump)

    tlistNano = np.concatenate([
        np.linspace(0, tRamp, nRamp, endpoint=False),
        np.linspace(tRamp, tRamp + tWait, nWait, endpoint=False),
        np.linspace(tRamp + tWait, tRamp + tWait + tJump, nJump, endpoint=False),
        np.linspace(tRamp + tWait + tJump, tTotal, nWait2)
    ])
    tlist = nsToMeV * tlistNano

    eiStart = 0.0
    eiSharp = 2.0 * fixedParameters[DQDParameters.U0.value]
    eiFinal = eiSharp

    eiValues = np.concatenate([
        np.linspace(eiStart, eiMiddle, nRamp, endpoint=False),
        np.full(nWait, eiMiddle),
        np.linspace(eiMiddle, eiSharp, nJump, endpoint=False),
        np.full(nWait2, eiFinal)
    ])

    rho0 = np.zeros((N, N))
    rho0[inverseCorrespondence[initialStateLabel], inverseCorrespondence[initialStateLabel]] = 1
    rho0 = Qobj(rho0)

    hEffList = []
    for ei in eiValues:
        params = fixedParameters.copy()
        params[DQDParameters.E_I.value] = ei

        H_full = dqd.project_hamiltonian(basis, parameters_to_change=params)
        H00 = H_full[:N, :N]
        H01 = H_full[:N, N:]
        H10 = H_full[N:, :N]
        H11 = H_full[N:, N:]

        hEff = H00 - H01 @ inv(H11) @ H10
        hEffList.append(Qobj(hEff))

    def hEffTimeDependent(t, args):
        idx = np.argmin(np.abs(tlist - t))
        return hEffList[idx]

    result = mesolve(hEffTimeDependent, rho0, tlist, c_ops=[])
    finalPop = result.states[-1].diag()

    totalPopulation = (
        finalPop[inverseCorrespondence["LL,S,T-"]]
        + finalPop[inverseCorrespondence["LR,S,T-"]]
        + finalPop[inverseCorrespondence["LR,T+,T-"]]
        + finalPop[inverseCorrespondence["LR,T0,T-"]]
    )

    I = (
        finalPop[inverseCorrespondence["LL,S,T-"]]
        + finalPop[inverseCorrespondence["LR,S,T-"]]
    ) / totalPopulation

    return np.real(I)


def run2DSweepOverTWaitAndDetuning(fixedParameters, initialStateLabel, tWaitList, eiMiddleList, totalPoints=500):
    nCores = multiprocessing.cpu_count()
    print(f"Ejecutando barrido 2D con {nCores} núcleos...")

    # Lista de pares (tWait, eiMiddle)
    paramPairs = [(t, ei) for ei in eiMiddleList for t in tWaitList]

    results = Parallel(n_jobs=nCores)(
        delayed(simulateCurrentAtTWait)(fixedParameters, initialStateLabel, tWait, eiMiddle, totalPoints)
        for (tWait, eiMiddle) in paramPairs
    )

    # Convertimos a matriz 2D
    currentMatrix = np.array(results).reshape(len(eiMiddleList), len(tWaitList))

    # === Gráfica 2D ===
    plt.figure(figsize=(8, 6))
    extent = [tWaitList[0], tWaitList[-1], eiMiddleList[0], eiMiddleList[-1]]
    plt.imshow(currentMatrix, aspect='auto', origin='lower', extent=extent, cmap='viridis')
    plt.colorbar(label='Current (normalized)')
    plt.xlabel("tWait (ns)")
    plt.ylabel("eiMiddle (meV)")
    plt.title("Current vs tWait and Detuning")
    plt.grid(False)

    # Guardar
    outputDir = os.path.join(os.getcwd(), "DQD_2particles_1orbital", "figures")
    os.makedirs(outputDir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    figPath = os.path.join(outputDir, f"Current_vs_tWait_eiMiddle_2D_{timestamp}.png")
    plt.savefig(figPath)
    print(f"Figura 2D guardada en: {figPath}")

    dataPath = os.path.join(outputDir, f"Current_vs_tWait_eiMiddle_2D_{timestamp}.npy")
    np.save(dataPath, currentMatrix)
    print(f"Matriz de corriente guardada en: {dataPath}")
    plt.show()

# === Parámetros ===
gOrtho = 10
U0 = 8.5
U1 = 0.1
fixedParameters = {
    DQDParameters.B_FIELD.value: 0.20,
    DQDParameters.B_PARALLEL.value: 0.15,
    DQDParameters.E_I.value: 0.0,
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

tWaitValues = np.linspace(0.1, 1.0, 10)  # ns
eiMiddleValues = np.linspace(7.5, 9.0, 10)  # meV

run2DSweepOverTWaitAndDetuning(fixedParameters, initialStateLabel="LR,T+,T-", tWaitList=tWaitValues, eiMiddleList=eiMiddleValues)
