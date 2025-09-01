import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from qutip import Qobj, Bloch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.DynamicsManager import DynamicsManager, DQDParameters, setupLogger


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from qutip import Qobj, Bloch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import sys
import os
import logging
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.DynamicsManager import DynamicsManager, DQDParameters, setupLogger


def densityToSTQubit(rho4, iSym, iAnti):
    """Project a 4x4 density matrix onto {|S>,|T>} qubit basis."""
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


def rho2ToBloch(rho2):
    """Return Bloch vector (sx, sy, sz)."""
    rho01 = rho2[0, 1]
    sx = 2*np.real(rho01)
    sy = -2*np.imag(rho01)
    sz = np.real(rho2[0, 0] - rho2[1, 1])
    return np.array([sx, sy, sz], dtype=float)


def animateSTQubit(result, tlistNano, iSym, iAnti, eiValues, DM,
                   outFile="st_qubit.mp4", cutOffN=None):
    """Animate Bloch vector and show static plots alongside it."""

    # --- Precompute Bloch vectors
    rhos = result.states if hasattr(result, "states") else result
    blochVectors = []
    for rho in rhos:
        rho2 = densityToSTQubit(rho, iSym, iAnti)
        blochVectors.append(rho2ToBloch(rho2))
    blochVectors = np.array(blochVectors)

    # --- Compute populations
    populations = np.array([state.diag() for state in rhos])
    statesToPlot = (
        [DM.correspondence[i] for i in range(cutOffN)]
        if cutOffN is not None else [DM.correspondence[i] for i in range(4)]
    )

    # Qubit populations
    sumTriplet = (
        populations[:, DM.invCorrespondence["LR,T0,T-"]] +
        populations[:, DM.invCorrespondence["LR,T-,T-"]]
    )
    sumSinglet = (
        populations[:, DM.invCorrespondence["LL,S,T-"]] +
        populations[:, DM.invCorrespondence["LR,S,T-"]]
    )
    sum4States = sumTriplet + sumSinglet

    # --- Create figure with 3 rows, 2 columns
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.3, 1.0])  # achicar fila del medio

    ax1 = fig.add_subplot(gs[0, 0])  # Poblaciones individuales
    ax_legend = fig.add_subplot(gs[1, 0])  # Solo para leyenda
    ax2 = fig.add_subplot(gs[2, 0])  # Sweep detuning
    ax3 = fig.add_subplot(gs[0, 1])  # Poblaciones qubit
    ax4 = fig.add_subplot(gs[1:, 1], projection="3d")  # Bloch sphere ocupa dos filas

    # --- Plot static curves
    # Poblaciones individuales
    lines = []
    labels = []
    for label in statesToPlot:
        index = DM.invCorrespondence[label]
        line, = ax1.plot(tlistNano, populations[:, index], label=label)
        lines.append(line)
        labels.append(label)
    ax1.set_ylabel('Population')
    ax1.set_title("Individual populations")
    ax1.grid()

    # Leyenda debajo
    ax_legend.axis("off")
    ax_legend.legend(lines, labels, loc="center", ncol=2)

    # Sweep detuning
    ax2.plot(tlistNano, eiValues, color='black', linewidth=2)
    ax2.set_ylabel(r'$E_I$ (meV)')
    ax2.set_xlabel("Time (ns)")
    ax2.set_title('Detuning sweep')
    ax2.grid()

    # Qubit populations
    ax3.plot(tlistNano, sumTriplet, label='T (antisym)', linestyle='--', color='tab:blue')
    ax3.plot(tlistNano, sumSinglet, label='S (sym)', linestyle='--', color='tab:green')
    ax3.plot(tlistNano, sum4States, label='Total', linestyle='-', color='tab:red')
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Populations')
    ax3.set_title('Total populations in relevant subspace')
    ax3.legend()
    ax3.grid()

    # --- Vertical lines that will move with time
    vline1 = ax1.axvline(x=tlistNano[0], color="red", linestyle=":")
    vline2 = ax2.axvline(x=tlistNano[0], color="red", linestyle=":")
    vline3 = ax3.axvline(x=tlistNano[0], color="red", linestyle=":")

    # --- Bloch sphere setup
    b = Bloch(fig=fig, axes=ax4)
    b.vector_color = ["#1f77b4"]
    b.point_color = ["#2ca02c"]
    b.view = [-60, 30]
    b.zlabel = [r"$|S\rangle$", r"$|T\rangle$"]

    def drawBloch(i):
        b.clear()
        b.add_vectors(blochVectors[i])
        k = max(0, i-15)
        b.add_points(blochVectors[k:i+1].T)
        b.make_sphere()

    def update(i):
        drawBloch(i)
        vline1.set_xdata([tlistNano[i], tlistNano[i]])
        vline2.set_xdata([tlistNano[i], tlistNano[i]])
        vline3.set_xdata([tlistNano[i], tlistNano[i]])
        return ax4, vline1, vline2, vline3

    framesToSave = np.linspace(0, len(tlistNano)-1, 400, dtype=int)  # 300 frames en lugar de 1800
    ani = FuncAnimation(fig, update, frames=framesToSave, interval=80, blit=False)

    # --- Añadir timestamp al nombre del archivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root, ext = os.path.splitext(outFile)
    outFileTimestamped = f"{root}_{timestamp}{ext}"
    

    ani.save(outFileTimestamped.replace(".mp4", ".gif"), writer=PillowWriter(fps=15))

    plt.close(fig)
    logging.info(f"Saved animation to {outFileTimestamped}")



if __name__ == "__main__":
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

    DM = DynamicsManager(fixedParameters)
    setupLogger()

    expectedPeriod = 1.5416  # Expected period in ns
    interactionTime = expectedPeriod*3.25
    phaseRotationTime = 3 * expectedPeriod

    logging.info(f"Running detuning protocol for {interactionTime/expectedPeriod} T interaction...")
    intervalTimes = [2*expectedPeriod, interactionTime,
                         phaseRotationTime, 1.5*expectedPeriod]
    totalPoints = 1000
    runOptions = DM.getRunOptions(atol=1e-8, rtol=1e-6, nsteps=10000)
    T1 = 100000
    T2star = 100000
    activateDephasing = False
    activateSpinRelaxation = False
    cutOffN = None
    filter = False


    spinRelaxation = None
    dephasing = None
    if activateSpinRelaxation:
        spinRelaxation = DM.gammaFromTime(T1)
    if activateDephasing:
        dephasing = DM.gammaFromTime(T2star)

    DM.fixedParameters["DecoherenceTime"] = DM.decoherenceTime(T2star, T1)

    tlistNano, eiValues = DM.obtainGateZProtocolParameters(
            intervalTimes, totalPoints, interactionDetuning=interactionDetuning
        )

    result = DM.detuningProtocol(
        tlistNano, eiValues, filter=filter,
        dephasing=dephasing, spinRelaxation=spinRelaxation,
        cutOffN=cutOffN, runOptions=runOptions
    )

    logging.info("Detuning protocol completed.")

    iSym = [DM.invCorrespondence["LL,S,T-"], DM.invCorrespondence["LR,S,T-"]]
    iAnti = [DM.invCorrespondence["LR,T0,T-"], DM.invCorrespondence["LR,T-,T-"]]

    animateSTQubit(result, tlistNano, iSym, iAnti, eiValues, DM,
                       outFile=os.path.join(DM.figuresDir, 'bloch_animation.gif'),
                       cutOffN=cutOffN)