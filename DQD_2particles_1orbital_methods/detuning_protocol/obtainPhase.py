import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj
from joblib import Parallel, delayed, cpu_count
import os
import logging
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.DynamicsManager import DynamicsManager, DQDParameters, setupLogger

def phiFromP0(p0):
    return 2 * np.arccos(np.sqrt(p0))

def runSingleFactor(DM, expectedPeriod, interactionDetuning, phaseAccumulationTime, topValue):
    def computeSlopeTime(yInitial, yFinal, slope=0.5):
        #Fix any slope to 2 ns for each meV
        deltaY = yFinal - yInitial
        x = deltaY / slope
        return x

    peakDetuning = DM.fixedParameters[DQDParameters.U0.value]
    slopeTimeOutside = computeSlopeTime(interactionDetuning, peakDetuning)
    slopeTimeInside =  computeSlopeTime(interactionDetuning, topValue)
    phaseAccumulationTime = 1.5*expectedPeriod

    slopesShapes = [
        [peakDetuning, peakDetuning, phaseAccumulationTime/2.0],
        [peakDetuning, interactionDetuning, slopeTimeOutside],
        [interactionDetuning, interactionDetuning, 1.25 * expectedPeriod-0.10],
        [interactionDetuning,peakDetuning, slopeTimeInside],
        [peakDetuning, peakDetuning, phaseAccumulationTime],
        [peakDetuning, interactionDetuning, slopeTimeInside],
        [interactionDetuning, interactionDetuning, 1.75 * expectedPeriod-0.15],
        [interactionDetuning,peakDetuning, slopeTimeOutside],
        [peakDetuning, peakDetuning, phaseAccumulationTime/2.0]
    ]

    initialStateDet = None
    initialStateBParallel = None
    totalPoints = 1200
    runOptions = DM.getRunOptions(atol=1e-8, rtol=1e-6, nsteps=10000)
    T1 = 100000
    T2star = 100000
    activateDephasing = False
    activateSpinRelaxation = False
    cutOffN = None


    spinRelaxation = None
    dephasing = None
    if activateSpinRelaxation:
        spinRelaxation = DM.gammaFromTime(T1)
    if activateDephasing:
        dephasing = DM.gammaFromTime(T2star)

    DM.fixedParameters["DecoherenceTime"] = DM.decoherenceTime(T2star, T1)

    tlistNano, eiValues = DM.buildGenericProtocolParameters(slopesShapes, totalPoints)

    result = DM.detuningProtocol(
        tlistNano, eiValues,
        dephasing=dephasing, spinRelaxation=spinRelaxation,
        cutOffN=cutOffN, runOptions=runOptions, 
        initialStateDetuning=initialStateDet,
        initialStateField=initialStateBParallel
    )

    rhoFinal = result.states[-1]
    population = rhoFinal.diag()
    p1 = (
        population[DM.invCorrespondence["LR,T0,T-"]] +
        population[DM.invCorrespondence["LR,T-,T-"]]
    )
    p0 = (
        population[DM.invCorrespondence["LL,S,T-"]] +
        population[DM.invCorrespondence["LR,S,T-"]]
    )

    phi = phiFromP0(p0) * 180 / np.pi

    return p0, p1, phi

if __name__ == "__main__":
    interactionDetuning = 4.7638
    gOrtho = 10
    expectedPeriod = 1.5416

    fixedParameters = {
        DQDParameters.B_FIELD.value: 1.50,
        DQDParameters.B_PARALLEL.value: 0.1,
        DQDParameters.E_I.value: interactionDetuning,
        DQDParameters.T.value: 0.05,
        DQDParameters.DELTA_SO.value: 0.066,
        DQDParameters.DELTA_KK.value: 0.02,
        DQDParameters.T_SOC.value: 0.0,
        DQDParameters.U0.value: 6.0,
        DQDParameters.U1.value: 1.5,
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

    setupLogger()
    DM = DynamicsManager(fixedParameters)

    phaseAccumulationTimes = np.linspace(0.01*expectedPeriod, 2.0*expectedPeriod, 100) 
    topValues = np.linspace(interactionDetuning*1.001, DM.fixedParameters[DQDParameters.U0.value], 7)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    for tv in topValues:

        maxCores = min(24, cpu_count())
        logging.info(f"Running with topValue={tv}, using {maxCores} cores.")
        results = Parallel(n_jobs=maxCores)(delayed(runSingleFactor)(DM, expectedPeriod, interactionDetuning, f, tv) 
                                      for f in phaseAccumulationTimes)

        resultsP0, resultsP1, resultsPhi = zip(*results)
        xValues = phaseAccumulationTimes/expectedPeriod  # normalizado por expectedPeriod

        # --- Graficar curvas para cada topValue
        ax1.plot(xValues, resultsP0, "o-", label=f"P0 (detuning peak / int det ={tv/interactionDetuning})")
        ax1.plot(xValues, resultsP1, "o-", label=f"P1 (detuning peak / int det ={tv/interactionDetuning})")
        ax2.plot(xValues, resultsPhi, "s-", label=f"Phi (detuning peak / int det ={tv/interactionDetuning})")

        # --- FFT para obtener frecuencia y periodo dominante ---

        # señal: eliminamos el offset
        y = np.array(resultsPhi) - np.mean(resultsPhi)
        x = np.array(xValues)
        N = len(y)
        dt = x[1] - x[0]  # paso en eje x

        fftVals = np.fft.rfft(y)
        fftFreqs = np.fft.rfftfreq(N, d=dt)

        dominantFreq = fftFreqs[np.argmax(np.abs(fftVals))]
        dominantPeriod = 1 / dominantFreq if dominantFreq != 0 else np.inf

        logging.info(f"\n=== Análisis FFT ===")
        logging.info(f"Frecuencia dominante ≈ {dominantFreq:.4f} (1/unidades de xValues)")
        logging.info(f"Periodo dominante ≈ {dominantPeriod:.4f} (unidades de xValues)\n")

    ax1.set_xlabel("Accumulated time / period")
    ax1.set_ylabel("Final Population")
    ax1.legend()
    ax1.grid()

    ax2.set_xlabel("Accumulated time / period")
    ax2.set_ylabel("Phi (grad)")
    ax2.set_ylim(0, 180)
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    DM.saveResults(name="Phase_accumulation")
    logging.info(f"Simulations completed for all topValues.\n")


