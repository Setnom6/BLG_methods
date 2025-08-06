from src.DQD_2particles_1orbital import DQD_2particles_1orbital, DQDParameters
import matplotlib.pyplot as plt
from scipy.linalg import solve_sylvester
from qutip import *
import numpy as np
import os
from datetime import datetime

def schriefferWolff(H_full):
    N0 = 5
    N1 = 6
    N2 = 17
    H00 = H_full[:N0, :N0]
    
    H01 = H_full[:N0, N0:N0+N1]
    H10 = H_full[N0:N0+N1, :N0]
    H11 = H_full[N0:N0+N1, N0:N0+N1]

    H12 = H_full[N0:N0+N1, N0+N1:N0+N1+N2]
    H21 = H_full[N0+N1:N0+N1+N2, N0:N0+N1]
    H22 = H_full[N0+N1:N0+N1+N2, N0+N1:N0+N1+N2]

    S12 = solve_sylvester(H11, -H22, -H12)
    H11_eff = H11 + 0.5 * (H12 @ S12.conj().T + S12 @ H21)

    S01 = solve_sylvester(H00, -H11_eff, -H01)
    H00_eff = H00 + 0.5 * (H01 @ S01.conj().T + S01 @ H10)

    return H00_eff


def runDynamics(fixedParameters):

    # === Sweep protocol parameters (in ns) ===
    tRamp, tWait, tJump, tWait2 = 0.6, 0.85, 0.05, 1.0
    tTotal = tRamp + tWait + tJump + tWait2
    totalPoints = 500
    nRamp = int(tRamp * totalPoints / tTotal)
    nWait = int(tWait * totalPoints / tTotal)
    nJump = int(tJump * totalPoints / tTotal)
    nWait2 = int(tWait2 * totalPoints / tTotal)

    nsToMeV = 1519.29

    dqd = DQD_2particles_1orbital(fixedParameters)
    basis = dqd.singlet_triplet_reordered_basis
    correspondence = dqd.singlet_triplet_reordered_correspondence
    inverseCorrespondence = {v: k for k, v in correspondence.items()}

    # === Time vectors (in ns and meV⁻¹) ===
    tlistNano = np.concatenate([
        np.linspace(0, tRamp, nRamp, endpoint=False),
        np.linspace(tRamp, tRamp + tWait, nWait, endpoint=False),
        np.linspace(tRamp + tWait, tRamp + tWait + tJump, nJump, endpoint=False),
        np.linspace(tRamp + tWait + tJump, tTotal, nWait2)
    ])
    tlist = nsToMeV * tlistNano

    # === Detuning profile E_I(t) ===
    eiStart = 0.0
    eiMiddle = 8.23
    eiSharp = 2.0 * 8.5
    eiFinal = 2.0 * 8.5

    eiValues = np.concatenate([
        np.linspace(eiStart, eiMiddle, nRamp, endpoint=False),
        np.full(nWait, eiMiddle),
        np.linspace(eiMiddle, eiSharp, nJump),
        np.full(nWait2, eiFinal)
    ])
    assert len(eiValues) == len(tlist)

    # === Initial state ===
    # Compute H_eff at t = 0 and get its ground state
    ei0 = eiValues[0]
    params = fixedParameters.copy()
    params[DQDParameters.E_I.value] = ei0

    H_full = dqd.project_hamiltonian(basis, parameters_to_change=params)
    H00_eff = schriefferWolff(H_full)

    # Convertir a Qobj si usas QuTiP
    H00_eff_qobj = Qobj(H00_eff)

    _, evecs = H00_eff_qobj.eigenstates()
    psi0 = evecs[0]  # ground state
    rho0 = psi0 * psi0.dag()

    # === Precompute effective Hamiltonians ===
    hEffList = []
    for ei in eiValues:
        params = fixedParameters.copy()
        params[DQDParameters.E_I.value] = ei

        H_full = dqd.project_hamiltonian(basis, parameters_to_change=params)
        H00_eff = schriefferWolff(H_full)

        # Convertir a Qobj si usas QuTiP
        H00_eff_qobj = Qobj(H00_eff)
        hEffList.append(H00_eff_qobj)

    # === Time-dependent Hamiltonian function ===
    def hEffTimeDependent(t, args):
        idx = np.argmin(np.abs(tlist - t))
        return hEffList[idx]

    # === Time evolution ===
    result = mesolve(hEffTimeDependent, rho0, tlist, c_ops=[])
    populations = np.array([state.diag() for state in result.states])

    # === Plotting: individual populations, detuning, total singlet/triplet ===
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(10, 10), height_ratios=[3, 1, 1])

    # Individual populations
    statesToPlot = [correspondence[i] for i in range(N0)]
    for label in statesToPlot:
        index = inverseCorrespondence[label]
        ax1.plot(tlistNano, populations[:, index], label=label)
    ax1.set_ylabel('Population')
    ax1.set_title('Population dynamics (SWT with detuning sweep)')
    ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    ax1.grid()

    # Detuning E_I(t)
    ax2.plot(tlistNano, eiValues, color='black', linewidth=2)
    ax2.set_ylabel(r'$E_I$ (meV)')
    ax2.set_title('Detuning sweep')
    ax2.grid()

    # Singlet and triplet subspace populations
    sumTriplet = (
        populations[:, inverseCorrespondence["LR,T+,T-"]] +
        populations[:, inverseCorrespondence["LR,T0,T-"]] +
        populations[:, inverseCorrespondence["LR,T-,T-"]]
    )
    sumSinglet = (
        populations[:, inverseCorrespondence["LL,S,T-"]] +
        populations[:, inverseCorrespondence["LR,S,T-"]]
    )
    sum5States = sumTriplet + sumSinglet

    ax3.plot(tlistNano, sumTriplet, label='Triplet', linestyle='--', color='tab:blue')
    ax3.plot(tlistNano, sumSinglet, label='Singlet', linestyle='--', color='tab:green')
    ax3.plot(tlistNano, sum5States, label='Total (5 states)', linestyle='-', color='tab:red')
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Populations')
    ax3.set_title('Total populations in relevant subspace')
    ax3.legend()
    ax3.grid()

    plt.subplots_adjust(hspace=0.4, right=0.75)

    # === Save results ===
    figuresDir = os.path.join(os.getcwd(), "DQD_2particles_1orbital", "figures")
    os.makedirs(figuresDir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    figPath = os.path.join(figuresDir, f"TimeSweep_SWT_{timestamp}.png")
    plt.savefig(figPath)
    print(f"Figure saved at: {figPath}")

    paramPath = os.path.join(figuresDir, f"parameters_TimeSweep_SWT_{timestamp}.txt")
    with open(paramPath, 'w') as f:
        for key, value in fixedParameters.items():
            f.write(f"{key}: {value}\n")
    print(f"Parameters saved at: {paramPath}")

    plt.show()


# === PARAMETERS ===
gOrtho = 10
U0 = 8.5
U1 = 0.1
Ei = 8.23
fixedParameters = {
        DQDParameters.B_FIELD.value: 0.20,
        DQDParameters.B_PARALLEL.value: 0.14,
        DQDParameters.E_I.value: 8.23,
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

# === EXECUTION ===
runDynamics(fixedParameters)