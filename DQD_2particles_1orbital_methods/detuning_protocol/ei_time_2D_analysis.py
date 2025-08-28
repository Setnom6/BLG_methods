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
import time
from scipy.fft import fft, fftfreq

# ---------------- dynamics ----------------
def runDynamics(detuning, parameters, times, cutOffN, dephasing, spinRelaxation):
        params = deepcopy(parameters)
        params[DQDParameters.E_I.value] = detuning
        DM = DynamicsManager(params)
        populations =  DM.simpleTimeEvolution(times, cutOffN=cutOffN, dephasing=dephasing, spinRelaxation=spinRelaxation)
        return DM.getCurrent(populations)


# ---------------- freq estimation ----------------
def estimateRabiFrequency(signal, times, paddingFactor=4):
    """
    Estimates the dominant Rabi frequency in a 1D signal (currents vs time)
    using FFT with zero-padding and parabolic correction.
    Returns frequency in 1/ns.
    """
    signal = np.array(signal) - np.mean(signal)  # remove DC component
    N = len(signal)
    dt = times[1] - times[0]

    # Zero-padding to improve resolution
    N_padded = paddingFactor * N
    yf = fft(signal, n=N_padded)
    xf = fftfreq(N_padded, dt)[:N_padded // 2]
    spectrum = 2.0 / N * np.abs(yf[:N_padded // 2])

    # Ignore DC
    spectrum[0] = 0.0

    # Main peak
    i = np.argmax(spectrum)
    if 1 <= i < len(spectrum) - 1:
        # Parabolic interpolation around the maximum
        y0, y1, y2 = spectrum[i - 1], spectrum[i], spectrum[i + 1]
        dx = (y2 - y0) / (2 * (2 * y1 - y2 - y0))
        dominantFreq = xf[i] + dx * (xf[1] - xf[0])
    else:
        dominantFreq = xf[i]

    return dominantFreq  # in 1/ns

def computeFreqsForDetuningSweep(currents, detuningList, times):
    """
    Finds the Rabi frequencies for each detuning in a 2D current matrix.
    """
    n = len(detuningList)
    freqs = np.zeros(n)
    maxCurrents = np.zeros(n)
    for i in range(n):
        signal = currents[i, :]
        freqs[i] = estimateRabiFrequency(signal, times)
        maxCurrents[i] = np.max(np.abs(signal))
    return freqs, maxCurrents

# ---------------- detuning analysis for minimal Rabi frequency ----------------
def findCentralDetuning(currents, detuningList, times, minCurrentFraction=0.05):
    """
    Finds the detuning value where the Rabi frequency is minimized.
    """

    freqsNs, maxCurrents = computeFreqsForDetuningSweep(currents, detuningList, times)
    currentThreshold = minCurrentFraction * np.max(maxCurrents)

    # Sort indices by increasing frequency
    sortedIndices = np.argsort(freqsNs)

    # Select first index with sufficient current
    for idx in sortedIndices:
        if maxCurrents[idx] >= currentThreshold:
            bestIndex = idx
            break
    else:
        # If all are practically zero, take the first one anyway
        bestIndex = sortedIndices[0]

    return detuningList[bestIndex], freqsNs[bestIndex], freqsNs

# ---------------- sweet spots ----------------

def findSweetSpotsFromFreqs(currents, detuningList, times, minCurrentFraction=0.05, relGradThreshold=0.1):
    """
    Finds sweet spots in the detuning sweep based on frequency gradients.
    """
    freqs, maxCurrents = computeFreqsForDetuningSweep(currents, detuningList, times)
    grads = np.gradient(freqs, detuningList)
    absGrads = np.abs(grads)
    maxAbsGrad = np.max(absGrads) if np.max(absGrads) > 0 else 1.0
    gradMask = absGrads <= relGradThreshold * maxAbsGrad
    currentThreshold = minCurrentFraction * np.max(maxCurrents)
    currentMask = maxCurrents >= currentThreshold
    combinedMask = gradMask & currentMask
    sweetIndices = np.where(combinedMask)[0]
    if len(sweetIndices) == 0:
        minIdx = np.argmin(absGrads)
        sweetIndices = np.array([minIdx])
    return sweetIndices, grads

def computeDephasingRateFromGradient(grads, sigmaEpsilon):
    return (grads ** 2) * (sigmaEpsilon ** 2)

# MonteCarlo con paralelización
def monteCarloFreqVarianceAtDetuning(detuningValue, parameters, times, cutOffN, dephasing, spinRelaxation,
                                     nSamples=200, sigmaEpsilon=0.01):
    def singleSample():
        epsSample = detuningValue + np.random.normal(0, sigmaEpsilon)
        params = deepcopy(parameters)
        params[DQDParameters.E_I.value] = epsSample
        DM = DynamicsManager(params)
        pops = DM.simpleTimeEvolution(times, cutOffN=cutOffN, dephasing=dephasing, spinRelaxation=spinRelaxation)
        sig = DM.getCurrent(pops)
        return estimateRabiFrequency(sig, times)
    numCores = min(cpu_count(), 16)
    freqs = Parallel(n_jobs=numCores)(delayed(singleSample)() for _ in range(nSamples))
    return np.mean(freqs), np.var(freqs)

# ---------------- frequency formatting ----------------
def formatFrequencies(freqsGHz):
    """
    Selects an appropriate prefix (Hz, kHz, MHz, GHz) and scales the frequencies.
    Returns:
        freqsScaled: array of scaled frequencies
        unit: string of the chosen unit
    """
    maxFreq = np.max(freqsGHz)
    
    if maxFreq >= 1:  # GHz o mayor
        return freqsGHz, "GHz"
    elif maxFreq >= 1e-3:  # MHz o mayor (pero menos de 1 GHz)
        return freqsGHz * 1e3, "MHz"
    elif maxFreq >= 1e-6:  # kHz o mayor (pero menos de 1 MHz)
        return freqsGHz * 1e6, "kHz"
    else:  # Hz (menos de 1 kHz)
        return freqsGHz * 1e9, "Hz"


if __name__ == "__main__":
    setupLogger()

    gOrtho = 10
    interactionDetuning = 4.7638  # Interaction detuning in meV
    fixedParameters = {
                DQDParameters.B_FIELD.value: 1.50,
                DQDParameters.B_PARALLEL.value: 0.1,
                DQDParameters.E_I.value: 0.0,
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
                DQDParameters.J.value: 0.00075 / gOrtho,}



    maxTime = 5.0
    totalPoints = 600
    cutOffN = None
    dephasing = None
    spinRelaxation = None

    detuningList = np.linspace(4.0, 6.0, totalPoints)

    parameterToChange = DQDParameters.B_PARALLEL.value
    arrayOfParameters = np.linspace(0.001, 1.0, 30)

    timesNs = np.linspace(0, maxTime, totalPoints)

    maxCores = 24
    availableCores = cpu_count()
    numCores = min(maxCores, availableCores)

    logging.info(f"Using {numCores} cores with joblib.")

    symmetryAxes = []
    rabiFreqs_sym = []
    rabiPeriods_sym = []
    for idx, value in enumerate(arrayOfParameters):
        parameters = deepcopy(fixedParameters)
        parameters[parameterToChange] = value
        currents = Parallel(n_jobs=numCores)(
            delayed(runDynamics)(detuning, parameters, timesNs, cutOffN, dephasing, spinRelaxation)
            for detuning in detuningList
        )
        currents = np.array(currents)  # por si joblib devuelve lista

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

        title += f" ({parameterToChange} = {value:.4f})"

        plt.title(title)

        DM = DynamicsManager(parameters)

        symDetuning, rabiFreq, freqsNs = findCentralDetuning(currents, detuningList, timesNs)
        sweetIndices, grads = findSweetSpotsFromFreqs(currents, detuningList, timesNs)
        symmetryAxes.append(symDetuning)

        plt.axhline(symDetuning, color='red', linestyle='--', label=f"Central detuning: {symDetuning:.4f} meV")
        rabiFreqs_sym.append(rabiFreq)
        rabiPeriods_sym.append(1.0 / rabiFreq) 
        logging.info(f"Central detuning (slowest Rabi) = {symDetuning:.4f} for {parameterToChange} = {value:.4f}")
        logging.info(f"Rabi frequency = {rabiFreq:.4f} 1/ns for {parameterToChange} = {value:.4f}")
        logging.info(f"Rabi period = {1/rabiFreq:.4f} ns for {parameterToChange} = {value:.4f}")

        logging.info(f"Simulation {idx+1}/{len(arrayOfParameters)} completed.\n")

        plt.legend()
        DM.saveResults(name="rabi_2D_ei")
        plt.close()

        # Sweet spots analysis

        fig, ax1 = plt.subplots(figsize=(7,4))

        ax1.plot(detuningList, freqsNs, 'o-', label='Rabi freq (1/ns)', color='tab:blue', markersize = 5)
        ax1.scatter(detuningList[sweetIndices], freqsNs[sweetIndices], c='red', zorder=5, s = 5, label='sweet spots')
        ax1.axvline(symDetuning, color='green', linestyle='--', label=f"Central detuning: {symDetuning:.4f} meV")
        ax1.set_xlabel('E_i (meV)')
        ax1.set_ylabel('Frequency (1/ns)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(detuningList, np.abs(grads), 'o-', label='|df/dε|', color='tab:orange', markersize = 5)
        ax2.scatter(detuningList[sweetIndices], np.abs(grads)[sweetIndices], s = 5, c='red', zorder=4)
        ax2.set_ylabel('|df/dε| (1/ns per meV)', color='tab:orange')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.title(f'Rabi frequency and sensitivity vs detuning for {parameterToChange} = {value:.4f}')
        plt.tight_layout()
        plt.savefig(os.path.join(DM.figuresDir, f"freq_and_sensitivity_vs_detuning_{value:.4f}.png"))
        plt.close()

        time.sleep(0.1)
    

    # --- Finalmente: graficar resultado eje vs bx

    rabiFreqs_sym = np.array(rabiFreqs_sym)
    freqsScaled, unit = formatFrequencies(rabiFreqs_sym)
    plt.figure(figsize=(8, 5))
    plt.plot(arrayOfParameters, rabiFreqs_sym, "o-", label="Rabi frequency central detuning")
    plt.xlabel(f"{parameterToChange}")
    plt.ylabel(f"Frequency ({unit})")
    plt.title("Rabi frequency vs {parameterToChange}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    figPath = os.path.join(DM.figuresDir, f"rabi_freq_vs_{parameterToChange}.png")
    plt.savefig(figPath)
    plt.close()

    periodsNs_sym = np.array(rabiPeriods_sym)
    plt.figure(figsize=(8,5))
    plt.plot(arrayOfParameters, periodsNs_sym, "o-", label="Rabi period central detuning")
    plt.xlabel(f"{parameterToChange}")
    plt.ylabel("Period (ns)")
    plt.title(f"Rabi period vs {parameterToChange}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    figPath = os.path.join(DM.figuresDir, f"rabi_period_vs_{parameterToChange}.png")
    plt.savefig(figPath)
    plt.close()


    symmetryAxes = np.array(symmetryAxes)
    plt.figure(figsize=(8, 5))
    plt.plot(arrayOfParameters, symmetryAxes, "o-", label="Detuning at interaction maxima (central detuning)")
    plt.xlabel(f"{parameterToChange}")
    plt.ylabel("Detuning (meV)")
    plt.title(f"Detuning at interaction maxima vs {parameterToChange}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    figPath = os.path.join(DM.figuresDir, f"detunings_vs_{parameterToChange}.png")
    plt.savefig(figPath)
    plt.close()

    paramPath = os.path.join(DM.figuresDir, f"parameters.txt")
    with open(paramPath, 'w') as f:
        for key, value in DM.fixedParameters.items():
            if key == parameterToChange or key == DQDParameters.E_I.value:
                continue
            else:
                f.write(f"{key}: {value}\n")
    print(f"Parameters saved at: {paramPath}")