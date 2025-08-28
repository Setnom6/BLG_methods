import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import logging
from joblib import Parallel, delayed

# Añadir path relativo al src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.DynamicsManager import DynamicsManager, DQDParameters, setupLogger


def runSingleSimulation(idx, interactionTime, fixedParameters, expectedPeriod,
                        totalPoints, interactionDetuning, inverseProtocol,
                        filter, dephasing, spinRelaxation, cutOffN, runOptions):
    """
    Ejecuta una sola simulación creando su propio DynamicsManager.
    """

    DM = DynamicsManager(fixedParameters)

    intervalTimes = [5*expectedPeriod, interactionTime, 5*expectedPeriod, 5*expectedPeriod]

    if inverseProtocol:
        tlistNano, eiValues = DM.obtainInverseProtocolParameters(
            intervalTimes, totalPoints, interactionDetuning=interactionDetuning
        )
    else:
        tlistNano, eiValues = DM.obtainOriginalProtocolParameters(
            intervalTimes, totalPoints, interactionDetuning=interactionDetuning
        )

    populations = DM.detuningProtocol(
        tlistNano, eiValues, filter=filter,
        dephasing=dephasing, spinRelaxation=spinRelaxation,
        cutOffN=cutOffN, runOptions=runOptions
    )

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

    gOrtho = 10
    interactionDetuning = 4.7638  # Interaction detuning in meV
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

    DMref = DynamicsManager(fixedParameters)

    expectedPeriod = 1.4862  # Expected period in ns
    totalPoints = 600
    runOptions = DMref.getRunOptions(atol=1e-8, rtol=1e-6, nsteps=10000)
    T1 = 10   # Spin relaxation time in ns
    T2star = 10  # Dephasing time in ns
    activateDephasing = True
    activateSpinRelaxation = True
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
    nPoints = 3
    interactionTimesToScan = np.linspace(expectedPeriod*4.5, expectedPeriod*5.5, nPoints)

    # ---------------- PARALLEL EXECUTION ----------------
    results = Parallel(n_jobs=-1)(
        delayed(runSingleSimulation)(
            idx, interactionTime, fixedParameters, expectedPeriod, totalPoints,
            interactionDetuning, inverseProtocol, filter, dephasing,
            spinRelaxation, cutOffN, runOptions
        )
        for idx, interactionTime in enumerate(interactionTimesToScan)
    )
    # -----------------------------------------------------

    finalPopulations, sumSinglet_list, sumTriplet_list = zip(*results)
    finalPopulations = np.array(finalPopulations)
    sumSinglet_list = np.array(sumSinglet_list)
    sumTriplet_list = np.array(sumTriplet_list)

    # Convertir interactionTimesToScan a múltiplos del periodo esperado
    interactionTimesMultiples = interactionTimesToScan / expectedPeriod

    # Graficar dos figuras una encima de la otra, eje temporal común
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Primera gráfica: sumas de singlete y triplete
    ax1.plot(interactionTimesMultiples, sumSinglet_list, marker='o', label='Singlet (current)', color='blue')
    ax1.plot(interactionTimesMultiples, sumTriplet_list, marker='o', label='Triplet (blockade)', color='orange')
    ax1.set_ylabel('Final population')
    ax1.set_title('Sensitivity of final Singlet and Triplet populations vs interaction time')
    ax1.legend()
    ax1.grid()

    # Segunda gráfica: poblaciones individuales
    statesToPlot = [DMref.correspondence[i] for i in range(4)]
    for label in statesToPlot:
        index = DMref.invCorrespondence[label]
        ax2.plot(interactionTimesMultiples, finalPopulations[:, statesToPlot.index(label)],
                 marker='o', label=label)
    ax2.set_xlabel('Interaction time / expected period')
    ax2.set_ylabel('Final population')
    ax2.set_title('Sensitivity of final population vs interaction time')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(DMref.figuresDir, 'sensitivity_interaction_time.png'))

