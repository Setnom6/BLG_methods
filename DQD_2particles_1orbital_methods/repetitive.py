import logging
from src.DQD_2particles_1orbital import DQD_2particles_1orbital, DQDParameters
import matplotlib.pyplot as plt
from scipy.linalg import inv, solve_sylvester
from qutip import *
import numpy as np
import os
from datetime import datetime
import multiprocessing
from copy import deepcopy
from functools import partial

def correctH11(H11, H12, H22):
    """Corrección de primer orden a H11 por acoplo con H22 usando el resolvente."""
    correction = H12 @ H22 @ H12.conj().T
    return H11 + correction

def schriefferWolff(H00, H01, H10, H11):
    """
    Aplica Schrieffer-Wolff para obtener H00_eff:
    Resuelve la ecuación de Sylvester para S01 y calcula H00_eff.
    """
    S01 = solve_sylvester(H00, -H11, -H01)
    H00_eff = H00 + 0.5 * (H01 @ S01.conj().T + S01 @ H10)
    return H00_eff

# === Setup logger ===
def setupLogger(logDir):
    os.makedirs(logDir, exist_ok=True)
    logPath = os.path.join(logDir, "error_log.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logPath),
            logging.StreamHandler()
        ]
    )

# Variable global que se inicializa una vez por proceso worker
globalDqd = None

def initWorker(fixedParameters):
    global globalDqd
    globalDqd = DQD_2particles_1orbital(fixedParameters)

def runDynamics(detuning, fixedParameters, tTotal=2.5, totalPoints=300):
    try:
        global globalDqd
        nsToMeV = 1519.29

        N0 = 5       # Tamaño subespacio 0
        N1 = 6       # Tamaño subespacio 1
        N2 = 17      # Tamaño subespacio 2

        basis = globalDqd.singlet_triplet_reordered_basis
        correspondence = globalDqd.singlet_triplet_reordered_correspondence
        inverseCorrespondence = {v: k for k, v in correspondence.items()}

        tlistNano = np.linspace(0, tTotal, totalPoints)
        tlist = nsToMeV * tlistNano

        # Estado inicial a zero detuning (se puede guardar una vez y pasar como parámetro, pero por simplicidad se recalcula)
        params_initial = deepcopy(fixedParameters)
        params_initial[DQDParameters.E_I.value] = 0.0

        H_full_initial = globalDqd.project_hamiltonian(basis, parameters_to_change=params_initial)
        H00 = H_full_initial[:N0, :N0]
        H01 = H_full_initial[:N0, N0:N0+N1]
        H10 = H_full_initial[N0:N0+N1, :N0]
        H11 = H_full_initial[N0:N0+N1, N0:N0+N1]

        H12 = H_full_initial[N0:N0+N1, N0+N1:N0+N1+N2]
        H21 = H_full_initial[N0+N1:N0+N1+N2, N0:N0+N1]
        H22 = H_full_initial[N0+N1:N0+N1+N2, N0+N1:N0+N1+N2]

        # Corregir H11 con efecto de H22
        H11_mod_initial = schriefferWolff(H11, H12, H21, H22)

        # Calcular Hamiltoniano efectivo con SWT
        H00_eff_initial = schriefferWolff(H00, H01, H10, H11_mod_initial)

        # Convertir a Qobj si usas QuTiP
        hEffQobj_initial = Qobj(H00_eff_initial)
        _, evecs_initial = hEffQobj_initial.eigenstates()
        psi0 = evecs_initial[0]
        rho0 = psi0 * psi0.dag()

        # Hamiltoniano efectivo para el detuning actual
        params = deepcopy(fixedParameters)
        params[DQDParameters.E_I.value] = detuning

        H_full = globalDqd.project_hamiltonian(basis, parameters_to_change=params)
        H00 = H_full[:N0, :N0]
        H01 = H_full[:N0, N0:N0+N1]
        H10 = H_full[N0:N0+N1, :N0]
        H11 = H_full[N0:N0+N1, N0:N0+N1]

        H12 = H_full[N0:N0+N1, N0+N1:N0+N1+N2]
        H21 = H_full[N0+N1:N0+N1+N2, N0:N0+N1]
        H22 = H_full[N0+N1:N0+N1+N2, N0+N1:N0+N1+N2]

        # Corregir H11 con efecto de H22
        H11_mod = schriefferWolff(H11, H12, H21, H22)

        # Calcular Hamiltoniano efectivo con SWT
        H00_eff = schriefferWolff(H00, H01, H10, H11_mod)

        # Convertir a Qobj si usas QuTiP
        hEffQobj = Qobj(H00_eff)

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

            I_t.append(I.real)

        logging.info(f"Simulation completed for detuning = {detuning:.3f} meV")
        return np.array(I_t)

    except Exception as e:
        logging.error(f"Error while simulating detuning = {detuning:.3f} meV: {e}")
        return np.zeros(300)

def plotCurrentMap(fixedParameters, detuningList, tTotal, totalPoints):
    maxCores = 4  # limitar núcleos a 4 (ajustar según servidor)
    availableCores = multiprocessing.cpu_count()
    numCores = min(maxCores, availableCores)

    logging.info(f"Using {numCores} cores for multiprocessing.")

    with multiprocessing.Pool(processes=numCores, initializer=initWorker, initargs=(fixedParameters,)) as pool:
        results = pool.map(partial(runDynamics, fixedParameters=fixedParameters, tTotal=tTotal, totalPoints=totalPoints), detuningList)

    currentMatrix = np.array(results)

    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        currentMatrix,
        aspect="auto",
        origin="lower",
        extent=[0, tTotal, detuningList[0], detuningList[-1]],
        cmap="viridis"
    )
    plt.colorbar(im, label="I (no Pauli Blockade)")
    plt.xlabel("Time (ns)")
    plt.ylabel("E_i (meV)")
    plt.title(f"Current vs detuning and interaction time for SWT," + 
              f"bx = {fixedParameters[DQDParameters.B_PARALLEL.value]:.3f} T, bz = {fixedParameters[DQDParameters.B_FIELD.value]:.3f} T")

    figuresDir = os.path.join(os.getcwd(), "DQD_2particles_1orbital", "figures", "forVideo")
    os.makedirs(figuresDir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    figPath = os.path.join(figuresDir, f"Current_ei_time_SWT_{timestamp}.png")
    plt.savefig(figPath)
    logging.info(f"Figure saved at: {figPath}")

    paramPath = os.path.join(figuresDir, f"parameters_Current_ei_time_SWT_{timestamp}.txt")
    with open(paramPath, 'w') as f:
        for key, value in fixedParameters.items():
            f.write(f"{key}: {value}\n")
    logging.info(f"Parameters saved at: {paramPath}")

if __name__ == "__main__":
    figuresDir = os.path.join(os.getcwd(), "DQD_2particles_1orbital", "figures", "forVideo")
    setupLogger(figuresDir)

    logging.info("Starting simulation...")

    bxValues = np.linspace(0.0, 0.7, 16)
    for bx in bxValues:
        gOrtho = 10
        U0 = 8.5
        U1 = 0.1
        fixedParameters = {
            DQDParameters.B_FIELD.value: 0.20,
            DQDParameters.B_PARALLEL.value: bx,
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

        totalTime = 2.5  # ns
        totalPoints = 300
        eiMiddleValues = np.linspace(7.9, 8.5, totalPoints)  # meV

        logging.info("Launching current map computation...")
        plotCurrentMap(fixedParameters, eiMiddleValues, totalTime, totalPoints)
        logging.info(f"Simulation completed for bx={bx:.2f} T.")







