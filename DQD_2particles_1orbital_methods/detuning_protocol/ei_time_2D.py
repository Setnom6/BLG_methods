import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.DynamicsManager import DynamicsManager, DQDParameters
import numpy as np
import matplotlib.pyplot as plt
import logging
import numpy as np
import os
from joblib import Parallel, delayed, cpu_count
from copy import deepcopy

def setupLogger():
        DM = DynamicsManager({})
        logDir = DM.figuresDir
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

def runDynamics(detuning, parameters, times, cutOffN, dephasing, spinRelaxation):
        params = deepcopy(parameters)
        params[DQDParameters.E_I.value] = detuning
        DM = DynamicsManager(params)
        populations =  DM.simpleTimeEvolution(times, cutOffN=cutOffN, dephasing=dephasing, spinRelaxation=spinRelaxation)
        return DM.getCurrent(populations)


if __name__ == "__main__":
    setupLogger()

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

    maxTime = 2.5
    totalPoints = 300
    cutOffN = None
    dephasing = None
    spinRelaxation = None

    detuningList = np.linspace(7.5, 9.5, totalPoints)
    bxList = np.array([0.179])

    timesNs = np.linspace(0, maxTime, totalPoints)

    maxCores = 24
    availableCores = cpu_count()
    numCores = min(maxCores, availableCores)

    logging.info(f"Using {numCores} cores with joblib.")

    for idx, bx in enumerate(bxList):
        parameters = deepcopy(fixedParameters)
        parameters[DQDParameters.B_PARALLEL.value] = bx
        currents = Parallel(n_jobs=numCores)(
            delayed(runDynamics)(detuning, parameters, timesNs, cutOffN, dephasing, spinRelaxation)
            for detuning in detuningList
        )

        plt.figure(figsize=(10, 6))
        im = plt.imshow(
            currents,
            aspect="auto",
            origin="lower",
            extent=[timesNs[0], timesNs[-1], detuningList[0], detuningList[-1]],
            cmap="viridis"
        )
        plt.colorbar(im, label="I (no Pauli Blockade)")
        plt.xlabel("Time (ns)")
        plt.ylabel("E_i (meV)")

        title = "Current vs detuning and interaction time"
        if cutOffN is not None:
            title += f" for {cutOffN} first states"
        else:
            title += " for SWT"

        if dephasing is not None:
            if spinRelaxation is not None:
                title += f", dephasing = {dephasing}, spin relaxation = {spinRelaxation}"
            else:
                title += f", dephasing = {dephasing}"
        else:
            if spinRelaxation is not None:
                title += f", spin relaxation = {spinRelaxation}"

        plt.title(title)
        DM = DynamicsManager(parameters)
        DM.saveResults(name="rabi_2D_ei")
        logging.info(f"Simulation {idx+1}/{len(bxList)} completed.\n")