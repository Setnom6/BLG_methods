import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from qutip import Bloch
import sys
import os
import logging
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.DynamicsManager import DynamicsManager, DQDParameters, setupLogger


def animateSTQubit(result, tlistNano, iSym, iAnti, eiValues, DM,
                   outFile="st_qubit.mp4", cutOffN=None,
                   nFrames=300, fps=25):
    """Animate Bloch vector and show static plots alongside it (optimizado)."""

    # --- Precompute Bloch vectors
    rhos = result.states if hasattr(result, "states") else result
    blochVectors = []
    for rho in rhos:
        rho2 = DM.densityToSTQubit(rho, iSym, iAnti)
        blochVectors.append(DM.rho2ToBloch(rho2))
    blochVectors = np.array(blochVectors)

    # --- Compute populations
    populations = np.array([state.diag() for state in rhos])
    statesToPlot = (
        [DM.correspondence[i] for i in range(cutOffN)]
        if cutOffN is not None else [DM.correspondence[i] for i in range(4)]
    )

    # Qubit populations
    sumSinglet, sumTriplet, sum4States = DM.getSingletTripletPopulations(populations, cutOff=cutOffN)

    # --- Compute ΔE_ST con la nueva función
    deltaE = DM.computeSingletTripletEnergyDifference(eiValues, cutOffN=cutOffN, iSym=iSym, iAnti=iAnti)


    # --- Create figure
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.3, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])  # Individual populations
    ax_legend = fig.add_subplot(gs[1, 0]) 
    ax2 = fig.add_subplot(gs[2, 0])  # Sweep detuning
    ax3 = fig.add_subplot(gs[0, 1])  # Qubit populations
    ax4 = fig.add_subplot(gs[1:, 1], projection="3d")  # Bloch sphere

    # --- Plot static curves
    lines, labels = [], []
    for label in statesToPlot:
        index = DM.invCorrespondence[label]
        line, = ax1.plot(tlistNano, populations[:, index], label=label)
        lines.append(line); labels.append(label)
    ax1.set_ylabel('Population'); ax1.set_title("Individual populations"); ax1.grid()
    ax_legend.axis("off"); ax_legend.legend(lines, labels, loc="center", ncol=2)

    # Eje izquierdo: detuning
    ax2.plot(tlistNano, eiValues, color='black', linewidth=2)
    ax2.set_ylabel(r'$E_I$ (meV)')
    ax2.set_xlabel("Time (ns)")
    ax2.set_title('Detuning sweep & ΔE_ST')
    ax2.grid()

    # Eje derecho: ΔE_ST
    ax2b = ax2.twinx()
    ax2b.plot(tlistNano, deltaE, color='tab:orange', linewidth=2, linestyle="--", label=r'$\Delta E_{ST}$')
    ax2b.set_ylabel(r'$\Delta E_{ST}$ (meV)', color='tab:orange')
    ax2b.tick_params(axis='y', labelcolor='tab:orange')

    # Leyenda combinada
    lines_ax2, labels_ax2 = ax2.get_legend_handles_labels()
    lines_ax2b, labels_ax2b = ax2b.get_legend_handles_labels()
    ax2b.legend(lines_ax2 + lines_ax2b, labels_ax2 + labels_ax2b, loc="upper right")

    ax3.plot(tlistNano, sumTriplet, label='T (antisym)', linestyle='--', color='tab:blue')
    ax3.plot(tlistNano, sumSinglet, label='S (sym)', linestyle='--', color='tab:green')
    ax3.plot(tlistNano, sum4States, label='Total', linestyle='-', color='tab:red')
    ax3.set_xlabel('Time (ns)'); ax3.set_ylabel('Populations')
    ax3.set_title('Total populations in relevant subspace')
    ax3.legend(); ax3.grid()

    # --- Vertical lines
    vline1 = ax1.axvline(x=tlistNano[0], color="red", linestyle=":")
    vline2 = ax2.axvline(x=tlistNano[0], color="red", linestyle=":")
    vline3 = ax3.axvline(x=tlistNano[0], color="red", linestyle=":")

    # --- Bloch sphere setup
    b = Bloch(fig=fig, axes=ax4)
    b.vector_color = ["#1f77b4"]; b.point_color = ["#2ca02c"]
    b.view = [-60, 30]; b.zlabel = [r"$|S\rangle$", r"$|T\rangle$"]

    def drawBloch(i):
        b.clear()
        b.add_vectors(blochVectors[i])
        k = max(0, i-10)  # solo últimos 10 puntos
        b.add_points(blochVectors[k:i+1].T)
        b.make_sphere()

    def update(i):
        drawBloch(i)
        vline1.set_xdata([tlistNano[i], tlistNano[i]])
        vline2.set_xdata([tlistNano[i], tlistNano[i]])
        vline3.set_xdata([tlistNano[i], tlistNano[i]])
        return []

    # --- Submuestreo de frames
    if nFrames < len(tlistNano):
        framesToSave = np.linspace(0, len(tlistNano)-1, nFrames, dtype=int)
    else:
        framesToSave = range(len(tlistNano))

    ani = FuncAnimation(fig, update, frames=framesToSave, interval=1000/fps, blit=False)

    # --- Guardar con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root, ext = os.path.splitext(outFile)
    outFileTimestamped = f"{root}_{timestamp}.mp4"

    ani.save(outFileTimestamped.replace(".mp4", ".gif"), writer=PillowWriter(fps=fps))
    plt.close(fig)
    logging.info(f"Saved animation to {outFileTimestamped}")


if __name__ == "__main__":
    gOrtho = 10
    interactionDetuning = 4.7638  # Interaction detuning in meV
    expectedPeriod = 1.5416  # Expected period in ns

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
    logging.info(f"Running detuning protocol interaction...")

    def computeSlopeTime(yInitial, yFinal, slope=0.5):
        #Fix any slope to 2 ns for each meV
        deltaY = yFinal - yInitial
        x = deltaY / slope
        return x

    peakDetuning = DM.fixedParameters[DQDParameters.U0.value]
    slopeTime = computeSlopeTime(interactionDetuning, peakDetuning)
    phaseAccumulationTime = 1.5*expectedPeriod

    slopesShapes = [
        [peakDetuning, peakDetuning, phaseAccumulationTime/2.0],
        [peakDetuning, interactionDetuning, slopeTime],
        [interactionDetuning, interactionDetuning, 1.25 * expectedPeriod-0.10],
        [interactionDetuning,peakDetuning, slopeTime],
        [peakDetuning, peakDetuning, phaseAccumulationTime],
        [peakDetuning, interactionDetuning, slopeTime],
        [interactionDetuning, interactionDetuning, 1.75 * expectedPeriod-0.15],
        [interactionDetuning,peakDetuning, slopeTime],
        [peakDetuning, peakDetuning, phaseAccumulationTime/2.0]
    ]

    initialStateDet = DM.fixedParameters[DQDParameters.U0.value]
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

    logging.info("Detuning protocol completed.")

    iSym = [DM.invCorrespondence["LL,S,T-"], DM.invCorrespondence["LR,S,T-"]]
    iAnti = [DM.invCorrespondence["LR,T0,T-"], DM.invCorrespondence["LR,T-,T-"]]

    animateSTQubit(result, tlistNano, iSym, iAnti, eiValues, DM,
                       outFile=os.path.join(DM.figuresDir, 'bloch_animation.gif'),
                       cutOffN=cutOffN)