import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import logging
from joblib import Parallel, delayed
from datetime import datetime

# Añadir path relativo al src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.DynamicsManager import DynamicsManager, DQDParameters, setupLogger


def runSingleSimulation(idx, interactionTime, slopesTime, fixedParameters, expectedPeriod,
                        totalPoints, interactionDetuning, inverseProtocol,
                        filter, dephasing, spinRelaxation, cutOffN, runOptions):
    """
    Ejecuta una sola simulación creando su propio DynamicsManager.
    """

    DM = DynamicsManager(fixedParameters)

    intervalTimes = [slopesTime, interactionTime-0.15, slopesTime, 5*expectedPeriod]

    if inverseProtocol:
        tlistNano, eiValues = DM.obtainInverseProtocolParameters(
            intervalTimes, totalPoints, interactionDetuning=interactionDetuning
        )
    else:
        tlistNano, eiValues = DM.obtainOriginalProtocolParameters(
            intervalTimes, totalPoints, interactionDetuning=interactionDetuning
        )

    result = DM.detuningProtocol(
        tlistNano, eiValues, filter=filter,
        dephasing=dephasing, spinRelaxation=spinRelaxation,
        cutOffN=cutOffN, runOptions=runOptions
    )

    populations = np.array([state.diag() for state in result.states])

    sumTriplet = (
        populations[-1, DM.invCorrespondence["LR,T0,T-"]] +
        populations[-1, DM.invCorrespondence["LR,T-,T-"]]
    )
    sumSinglet = (
        populations[-1, DM.invCorrespondence["LL,S,T-"]] +
        populations[-1, DM.invCorrespondence["LR,S,T-"]]
    )

    logging.info(f"Simulation {idx+1} completed.")

    return populations[-1, :], sumSinglet, sumTriplet


if __name__ == "__main__":
    setupLogger()
    logging.info("Starting simulations...")

    gOrtho = 10
    interactionDetuningExpected = 4.7638 # Interaction detuning in meV
    fixedParameters = {
        DQDParameters.B_FIELD.value: 1.50,
        DQDParameters.B_PARALLEL.value: 0.1,
        DQDParameters.E_I.value: interactionDetuningExpected,
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

    DMref = DynamicsManager(fixedParameters)

    expectedPeriod = 1.5416  # Expected period in ns
    totalPoints = 1000
    runOptions = DMref.getRunOptions(atol=1e-8, rtol=1e-6, nsteps=10000)
    T1 = 10   # Spin relaxation time in ns
    T2star = 10  # Dephasing time in ns
    activateDephasing = False
    activateSpinRelaxation = False
    cutOffN = None
    filter = False

    inverseProtocol = True  # If True, the protocol is inverted

    spinRelaxation = None
    dephasing = None
    if activateSpinRelaxation:
        spinRelaxation = DMref.gammaFromTime(T1)
    if activateDephasing:
        dephasing = DMref.gammaFromTime(T2star)

    DMref.fixedParameters["DecoherenceTime"] = DMref.decoherenceTime(T2star, T1)

    # Interaction times a explorar
    nPoints = 70
    interactionTime = expectedPeriod*5.5

    slopesToScan = np.linspace(0.1*expectedPeriod, 4*interactionTime, nPoints)

    # ---------------- PARALLEL EXECUTION ----------------
    results = Parallel(n_jobs=-1)(
        delayed(runSingleSimulation)(
            idx, interactionTime, slopeTime, fixedParameters, expectedPeriod, totalPoints,
            interactionDetuningExpected, inverseProtocol, filter, dephasing,
            spinRelaxation, cutOffN, runOptions
        )
        for idx, slopeTime in enumerate(slopesToScan)
    )
    # -----------------------------------------------------

    logging.info("Simulations finished. Creating graphs")
    finalPopulations, sumSinglet_list, sumTriplet_list = zip(*results)
    finalPopulations = np.array(finalPopulations)
    sumSinglet_list = np.array(sumSinglet_list)
    sumTriplet_list = np.array(sumTriplet_list)

    # Convertir interactionTimesToScan a múltiplos del periodo esperado
    interactionTimesMultiples = slopesToScan / expectedPeriod

    # Graficar dos figuras una encima de la otra, eje temporal común
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Primera gráfica: sumas de singlete y triplete
    ax1.plot(interactionTimesMultiples, sumSinglet_list, marker='o', label='Singlet (current)', color='blue')
    ax1.plot(interactionTimesMultiples, sumTriplet_list, marker='o', label='Triplet (blockade)', color='orange')
    ax1.axvline(x=1.0, color='gray', linestyle='--', linewidth=1)
    ax1.axvline(x=interactionTime/expectedPeriod, color='gray', linestyle='--', linewidth=1)
    ax1.set_ylabel('Final population')
    ax1.set_title('Sensitivity of final populations vs slopes time')
    ax1.legend()
    ax1.grid()

    # Segunda gráfica: poblaciones individuales
    statesToPlot = [DMref.correspondence[i] for i in range(4)]
    for label in statesToPlot:
        index = DMref.invCorrespondence[label]
        ax2.plot(interactionTimesMultiples, finalPopulations[:, statesToPlot.index(label)],
                 marker='o', label=label)
        

    ax2.axvline(x=1.0, color='gray', linestyle='--', linewidth=1)
    ax2.axvline(x=interactionTime/expectedPeriod, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Slopes time / T_exp')
    ax2.set_ylabel('Final population')
    ax2.set_title(f'bx = {DMref.fixedParameters[ DQDParameters.B_PARALLEL.value]:.2f} T, ER_exp plateau = {interactionDetuningExpected:.3f} meV, T_exp = {expectedPeriod:.3f} ns')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(DMref.figuresDir, f'sensitivity_slopes_time_{timestamp}.png'))
    DMref.saveResults(name="Detuning_protocol_iteration")