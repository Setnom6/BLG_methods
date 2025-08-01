from src.DQD_2particles_1orbital import DQD_2particles_1orbital, DQDParameters
from src.LindblandOperator import LindbladOperator
import matplotlib.pyplot as plt
from scipy.linalg import inv
from qutip import *
import numpy as np
import os
from datetime import datetime


def runDynamics(fixedParameters, initialStateLabel):

    # === Parámetros del protocolo de barrido ===
    tRamp, tWait, tJump, tWait2 = 0.6, 1.66, 0.6, 0.5  # en ns
    tTotal = tRamp+tWait+tJump+tWait2
    totalPoints = 500
    nRamp, nWait, nJump, nWait2 = int(tRamp*totalPoints/tTotal), int(tWait*totalPoints/tTotal), int(tJump*totalPoints/tTotal),  int(tWait2*totalPoints/tTotal)  # número de puntos por tramo

    nsToMeV = 1519.29
    dqd = DQD_2particles_1orbital(fixedParameters)
    basis = dqd.singlet_triplet_reordered_basis
    correspondence = dqd.singlet_triplet_reordered_correspondence
    inverseCorrespondence = {v: k for k, v in correspondence.items()}
    N = 5  # Tamaño del bloque H00

    # === Tiempo en ns y meV⁻¹ ===
    tlistNano = np.concatenate([
        np.linspace(0, tRamp, nRamp, endpoint=False),
        np.linspace(tRamp, tRamp + tWait, nWait, endpoint=False),
        np.linspace(tRamp + tWait, tRamp + tWait + tJump, nJump, endpoint=False),
        np.linspace(tRamp + tWait + tJump, tRamp + tWait + tJump+tWait2, nWait2)
    ])

    tlist = nsToMeV * tlistNano

    # === Detuning E_I(t) ===
    eiStart = 0.0
    eiMiddle = 8.25
    eiSharp = 2.0 * 8.5
    eiFinal = 2.0 * 8.5

    eiValues = np.concatenate([
        np.linspace(eiStart, eiMiddle, nRamp, endpoint=False),
        np.full(nWait, eiMiddle),
        np.linspace(eiMiddle, eiSharp, nJump),
        np.full(nWait2, eiFinal)
    ])
    assert len(eiValues) == len(tlist)

    # === Estado inicial ===
    rho0 = np.zeros((N, N))
    rho0[inverseCorrespondence[initialStateLabel], inverseCorrespondence[initialStateLabel]] = 1
    rho0 = Qobj(rho0)

    # === Precalcular Hamiltonianos efectivos ===
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

    # === Hamiltoniano dependiente del tiempo ===
    def hEffTimeDependent(t, args):
        idx = np.argmin(np.abs(tlist - t))
        return hEffList[idx]

    # === Evolución temporal ===
    result = mesolve(hEffTimeDependent, rho0, tlist, c_ops=[])

    populations = np.array([state.diag() for state in result.states])

    # === Gráfica con subplot adicional para E_I(t) ===
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 7), height_ratios=[3, 1])

    # Poblaciones
    statesToPlot = ["LR,T+,T-", "LL,S,T-", "LR,S,T-", 'LR,T0,T-', 'LR,T-,T-']
    for label in statesToPlot:
        index = inverseCorrespondence[label]
        ax1.plot(tlistNano, populations[:, index], label=label)
    ax1.set_ylabel('Population')
    ax1.set_title('Population dynamics (SWT with detuning sweep)')
    ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    ax1.grid()

    # Detuning
    ax2.plot(tlistNano, eiValues, color='black', linewidth=2)
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel(r'$E_I$ (meV)')
    ax2.set_title('Detuning sweep')
    ax2.grid()

    plt.subplots_adjust(hspace=0.3, right=0.75)

    # === Guardar resultados ===
    figuresDir = os.path.join(os.getcwd(), "DQD_2particles_1orbital", "figures")
    os.makedirs(figuresDir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    figPath = os.path.join(figuresDir, f"TimeSweep_{timestamp}.png")
    plt.savefig(figPath)
    print(f"Figura guardada en: {figPath}")

    paramPath = os.path.join(figuresDir, f"parameters_TimeSweep_{timestamp}.txt")
    with open(paramPath, 'w') as f:
        for key, value in fixedParameters.items():
            f.write(f"{key}: {value}\n")
    print(f"Parámetros guardados en: {paramPath}")

    plt.show()


# === PARÁMETROS ===
gOrtho = 10
U0 = 8.5
U1 = 0.1
Ei = 8.25
fixedParameters = {
    DQDParameters.B_FIELD.value: 0.2,
    DQDParameters.B_PARALLEL.value: 0.05,
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

# === EJECUTAR ===
runDynamics(fixedParameters, initialStateLabel="LR,T+,T-")



