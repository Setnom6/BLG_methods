import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.DynamicsManager import DynamicsManager, DQDParameters, setupLogger
import numpy as np
import matplotlib.pyplot as plt
import logging
import numpy as np
import os
from joblib import Parallel, delayed, cpu_count
from copy import deepcopy


def runDynamics(bx, parameters, times):
        params = deepcopy(parameters)
        params[DQDParameters.B_PARALLEL.value] = bx
        DM = DynamicsManager(params)
        populations =  DM.simpleTimeEvolution(times)
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
    bxList = np.linspace(0.15, 0.5, totalPoints)
    detuningList = np.linspace(7.9, 8.4, 20)
    

    timesNs = np.linspace(0, maxTime, totalPoints)

    maxCores = 24
    availableCores = cpu_count()
    numCores = min(maxCores, availableCores)

    logging.info(f"Using {numCores} cores with joblib.")

    for idx, ei in enumerate(detuningList):
        parameters = deepcopy(fixedParameters)
        parameters[DQDParameters.E_I.value] = ei
        currents = Parallel(n_jobs=numCores)(
            delayed(runDynamics)(bx, parameters, timesNs)
            for bx in bxList
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
        plt.ylabel("bx (T)")
        plt.title(f"Current vs bx and interaction time for SWT, " + 
                f"bx = {fixedParameters[DQDParameters.B_PARALLEL.value]:.3f} T, bz = {fixedParameters[DQDParameters.B_FIELD.value]:.3f} T")
        
        DM = DynamicsManager(parameters)
        DM.saveResults(name="rabi_2D_bx")
        logging.info(f"Simulation {idx+1}/{len(detuningList)} completed.\n")