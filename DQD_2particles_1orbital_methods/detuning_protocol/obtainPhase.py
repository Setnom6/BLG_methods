import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj
from joblib import Parallel, delayed, cpu_count
import os
import logging

from src.DynamicsManager import DynamicsManager, DQDParameters, setupLogger

def densityToSTQubit(rho4, iSym, iAnti):
    if isinstance(rho4, Qobj):
        rho = rho4.full()
    else:
        rho = np.asarray(rho4, dtype=complex)
    s, t = np.array(iSym), np.array(iAnti)

    rhoSS = np.trace(rho[np.ix_(s, s)])
    rhoTT = np.trace(rho[np.ix_(t, t)])
    rhoST = np.sum(rho[np.ix_(s, t)])
    rhoTS = np.conjugate(rhoST)

    rho2 = np.array([[rhoSS, rhoST],
                     [rhoTS, rhoTT]], dtype=complex)
    rho2 /= np.trace(rho2)
    return rho2

def phiFromP0(p0):
    return 2 * np.arccos(np.sqrt(p0))

def runSingleFactor(DM, expectedPeriod, interactionDetuning, factor, iSym, iAnti):
    slopesShapes = [
        [DM.fixedParameters[DQDParameters.U0.value], interactionDetuning, 1.0*expectedPeriod]
        [interactionDetuning, interactionDetuning, 0.25 * expectedPeriod],
        [interactionDetuning, DM.fixedParameters[DQDParameters.U0.value], factor * expectedPeriod],
        [DM.fixedParameters[DQDParameters.U0.value], DM.fixedParameters[DQDParameters.U0.value], factor * expectedPeriod],
        [DM.fixedParameters[DQDParameters.U0.value], interactionDetuning, factor * expectedPeriod],
        [interactionDetuning, interactionDetuning, 0.75 * expectedPeriod],
        [interactionDetuning, DM.fixedParameters[DQDParameters.U0.value], factor * expectedPeriod],
        [DM.fixedParameters[DQDParameters.U0.value], DM.fixedParameters[DQDParameters.U0.value], factor * expectedPeriod],
    ]

    totalPoints = 1200
    tlistNano, eiValues = DM.buildGenericProtocolParameters(slopesShapes, totalPoints)
    bValues = np.zeros_like(tlistNano)

    result = DM.combinedProtocol(
        tlistNano, bValues, eiValues,
        dephasing=None, spinRelaxation=None,
        cutOffN=None, runOptions=DM.getRunOptions()
    )

    rhoFinal = result.states[-1]
    rho2 = densityToSTQubit(rhoFinal, iSym, iAnti)
    p0 = np.real(rho2[0, 0])
    phi = phiFromP0(p0)

    return p0, phi

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

    iSym = [DM.invCorrespondence["LL,S,T-"], DM.invCorrespondence["LR,S,T-"]]
    iAnti = [DM.invCorrespondence["LR,T0,T-"], DM.invCorrespondence["LR,T-,T-"]]

    factors = np.linspace(0.01, 1.0, 20) 

    # --- Paralelizar la simulación
    maxCores = min(24, cpu_count())
    results = Parallel(n_jobs=maxCores)(delayed(runSingleFactor)(DM, expectedPeriod, interactionDetuning, f, iSym, iAnti) 
                                  for f in factors)

    resultsP0, resultsPhi = zip(*results)
    xValues = 6 * factors  # normalizado por expectedPeriod

    # --- Graficar
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.plot(xValues, resultsP0, "o-", label="Final P0")
    ax1.set_xlabel("6 * factor")
    ax1.set_ylabel("Final P0")
    ax1.legend()
    ax1.grid()

    ax2.plot(xValues, resultsPhi, "s-", color="tab:red", label="Phi")
    ax2.set_xlabel("6 * factor")
    ax2.set_ylabel("Phi (rad)")
    ax2.set_ylim(0, np.pi)
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()
