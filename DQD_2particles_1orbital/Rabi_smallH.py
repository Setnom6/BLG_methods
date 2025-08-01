from src.DQD_2particles_1orbital import DQD_2particles_1orbital, DQDParameters
from src.LindblandOperator import LindbladOperator
import matplotlib.pyplot as plt
from scipy.linalg import inv

from qutip import *
import numpy as np
import os
from datetime import datetime


def runDynamics(fixedParameters, initialStateLabel, useCollapseOperators=True):

    
    dqd = DQD_2particles_1orbital(fixedParameters)
    basis = dqd.singlet_triplet_reordered_basis
    correspondence = dqd.singlet_triplet_reordered_correspondence
    H_isolated = dqd.project_hamiltonian(basis, parameters_to_change=fixedParameters)

    N = 5
    H00 = H_isolated[:N, :N]
    H01 = H_isolated[:N, N:]
    H10 = H_isolated[N:, :N]
    H11 = H_isolated[N:, N:]

    # SWT effective Hamiltonian
    hEff = H00 - H01 @ inv(H11) @ H10

    inverseCorrespondence = {v: k for k, v in correspondence.items()}

    H_qutip = Qobj(hEff)

    if useCollapseOperators:
        lindbladCreator = LindbladOperator(dqd.FSU)
        listOfDissipators = []

        # Dephasing operators
        for i in range(8):
            LFockBasis = 0.05 * lindbladCreator.buildDephasingOperator(i)
            LProjected = dqd.project_hamiltonian(basis, alternative_operator=LFockBasis)
            listOfDissipators.append(Qobj(LProjected))

        # Relaxation operators
        relaxationsAllowed = [(0, 1), (2, 3), (4, 5), (6, 7)]
        for i, (k, l) in enumerate(relaxationsAllowed):
            LFockBasis = 0.05 * lindbladCreator.buildDecoherenceOperator(k, l)
            LProjected = dqd.project_hamiltonian(basis, alternative_operator=LFockBasis)
            listOfDissipators.append(Qobj(LProjected))
    else:
        listOfDissipators = []

    # Estado inicial
    rho0 = np.zeros_like(hEff)
    rho0[inverseCorrespondence[initialStateLabel], inverseCorrespondence[initialStateLabel]] = 1
    rho0 = Qobj(rho0)

    tlistNano = np.linspace(0, 1, 100) # in nanoseconds
    timeUnitsConversor = 1519.29 # As we use meV, then 1meV <-> 0.6582 ps ->  1 ns = 1000 ps = 1519.29 meV 
    tlist = timeUnitsConversor*tlistNano

    result = mesolve(H_qutip, rho0, tlist, c_ops=listOfDissipators)

    populations = np.array([state.diag() for state in result.states])

    # Gráfica
    plt.figure(figsize=(10, 6))
    statesToPlot = ["LR,T+,T-", "LL,S,T-", "LR,S,T-", 'LR,T0,T-', 'LR,T-,T-']
    for label in statesToPlot:
        index = inverseCorrespondence[label]
        plt.plot(tlistNano, populations[:, index], label=f'{label}')
    

    plt.xlabel('Time (ns)')
    plt.ylabel('Population')
    plt.title('Population dynamics' + (' with decoherence' if useCollapseOperators else ' (unitary evolution)'))
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.grid()
    plt.subplots_adjust(right=0.75)


    figures_dir = os.path.join(os.getcwd(),"DQD_2particles_1orbital", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    fig_path = os.path.join(figures_dir, f"Rabi_{timestamp}.png")
    plt.savefig(fig_path)
    print(f"Figure saved in: {fig_path}")

    param_path = os.path.join(figures_dir, f"parameters_Rabi_{timestamp}.txt")
    with open(param_path, 'w') as f:
        for key, value in fixedParameters.items():
            f.write(f"{key}: {value}\n")
    print(f"Parameters saved in: {param_path}")
    plt.show()


# === PARÁMETROS ===
gOrtho = 10
U0 = 8.5
U1 = 0.1
Ei = 8.25
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


# === LLAMADA A LA SIMULACIÓN ===
runDynamics(fixedParameters, initialStateLabel="LR,T+,T-", useCollapseOperators=False)