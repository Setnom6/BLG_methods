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


def runDynamics(detuning, parameters, times, cutOffN, dephasing, spinRelaxation, runOptions):
        params = deepcopy(parameters)
        params[DQDParameters.E_I.value] = detuning
        DM = DynamicsManager(params)
        populations =  DM.simpleTimeEvolution(times, cutOffN=cutOffN, dephasing=dephasing, spinRelaxation=spinRelaxation, runOptions=runOptions)
        return DM.getCurrent(populations)


if __name__ == "__main__":
    setupLogger()

    gOrtho = 10
    fixedParameters = {
                    DQDParameters.B_FIELD.value: 1.50,
                    DQDParameters.B_PARALLEL.value: 0.1586,
                    DQDParameters.E_I.value: 3.1839,
                    DQDParameters.T.value: 0.4,
                    DQDParameters.DELTA_SO.value: -0.04,
                    DQDParameters.DELTA_KK.value: 0.02,
                    DQDParameters.T_SOC.value: 0.0,
                    DQDParameters.U0.value: 10,
                    DQDParameters.U1.value: 5,
                    DQDParameters.X.value: 0.02,
                    DQDParameters.G_ORTHO.value: gOrtho,
                    DQDParameters.G_ZZ.value: 10 * gOrtho,
                    DQDParameters.G_Z0.value: 2 * gOrtho / 3,
                    DQDParameters.G_0Z.value: 2 * gOrtho / 3,
                    DQDParameters.GS.value: 2,
                    DQDParameters.GSLFACTOR.value: 1.0,
                    DQDParameters.GV.value: 28.0,
                    DQDParameters.GVLFACTOR.value: 0.66,
                    DQDParameters.A.value: 0.1,
                    DQDParameters.P.value: 0.02,
                    DQDParameters.J.value: 0.00075 / gOrtho,
        }

    maxTime = 20
    totalPoints = 1200
    T1 = 2e2  # Spin relaxation time in ns
    T2 = 1e2  # Dephasing time in ns
    activateDephasing = True
    activateSpinRelaxation = True
    cutOffN = None
    filter = False


    spinRelaxation = None
    dephasing = None
    DM = DynamicsManager(fixedParameters)
    runOptions = DM.getRunOptions(atol=1e-8, rtol=1e-6, nsteps=10000)

    if activateSpinRelaxation:
        spinRelaxation = DM.gammaRelaxation(T1)  # Spin relaxation time in meV
    if activateDephasing:
        dephasing = DM.gammaDephasing(T2, T1)  # Dephasing and spin relaxation time in meV

    detuningList = np.linspace(2.85, 3.85, totalPoints)
    bxList = np.array([0.1586])

    timesNs = np.linspace(0, maxTime, totalPoints)

    maxCores = 24
    availableCores = cpu_count()
    numCores = min(maxCores, availableCores)

    logging.info(f"Using {numCores} cores with joblib.")

    for idx, bx in enumerate(bxList):
        parameters = deepcopy(fixedParameters)
        parameters[DQDParameters.B_PARALLEL.value] = bx
        currents = Parallel(n_jobs=numCores)(
            delayed(runDynamics)(detuning, parameters, timesNs, cutOffN, dephasing, spinRelaxation, runOptions)
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

        if activateDephasing:
            if not activateSpinRelaxation:
                title += f" with T1 {T1:.3e} ns and T2 {T2:.3e} ns (just dephasing)"
            else:
                title += f" with T1 {T1:.3e} ns and T2 {T2:.3e} ns"

        else:
            if activateSpinRelaxation:
                title += f" with T1 {T1:.3e} ns (no dephasing)"

        plt.title(title)
        DM.saveResults(name="rabi_2D_ei")
        logging.info(f"Simulation {idx+1}/{len(bxList)} completed.\n")