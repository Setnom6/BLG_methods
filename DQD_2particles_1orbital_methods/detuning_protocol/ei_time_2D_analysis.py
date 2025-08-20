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
import time
from scipy.fft import fft, fftfreq

def setupLogger():
        DM = DynamicsManager({})
        logDir = DM.figuresDir
        os.makedirs(logDir, exist_ok=True)
        logPath = os.path.join(logDir, "log_results.txt")

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

def findCentralDetuning(currents, detuningList, times, minCurrentFraction=0.05):
    """
    Finds the detuning value where the Rabi frequency is minimized.
    """
    freqsNs = []
    maxCurrents = []

    for i in range(len(detuningList)):
        signal = currents[i, :]
        freq = estimateRabiFrequency(signal, times)  # in 1/ns
        freqsNs.append(freq)
        maxCurrents.append(np.max(np.abs(signal)))

    freqsNs = np.array(freqsNs)
    maxCurrents = np.array(maxCurrents)
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
    fixedParameters = {
                DQDParameters.B_FIELD.value: 1.50,
                DQDParameters.B_PARALLEL.value: 1.35,
                DQDParameters.E_I.value: 0.0,
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


    maxTime = 2.5
    totalPoints = 300
    cutOffN = None
    dephasing = None
    spinRelaxation = None

    detuningList = np.linspace(2.0, 7.0, totalPoints)

    parameterToChange = DQDParameters.t.value
    arrayOfParameters = np.linspace(0.0001, 0.5, 20)

    timesNs = np.linspace(0, maxTime, totalPoints)

    maxCores = 24
    availableCores = cpu_count()
    numCores = min(maxCores, availableCores)

    logging.info(f"Using {numCores} cores with joblib.")

    symmetryAxes = []
    rabiFreqs = []
    rabiPeriods = []
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
        DM.saveResults(name="rabi_2D_ei")

        symDetuning, rabiFreq, _ = findCentralDetuning(currents, detuningList, timesNs)
        symmetryAxes.append((value, symDetuning))
        rabiFreqs.append((value, rabiFreq))
        rabiPeriods.append((value, 1 / rabiFreq)) 
        logging.info(f"Central detuning (slowest Rabi) = {symDetuning:.4f} for {parameterToChange} = {value:.4f}")
        logging.info(f"Rabi frequency = {rabiFreq:.4f} 1/ns for {parameterToChange} = {value:.4f}")
        logging.info(f"Rabi period = {1/rabiFreq:.4f} ns for {parameterToChange} = {value:.4f}")

        logging.info(f"Simulation {idx+1}/{len(arrayOfParameters)} completed.\n")
        plt.close()
        time.sleep(0.1)
    

    # --- Finalmente: graficar resultado eje vs bx

    rabiFreqs = np.array(rabiFreqs)
    freqsScaled, unit = formatFrequencies(rabiFreqs[:,1])
    plt.figure(figsize=(8, 5))
    plt.plot(rabiFreqs[:,0], rabiFreqs[:,1], "o-", label="Rabi frequency")
    plt.xlabel(f"{parameterToChange}")
    plt.ylabel(f"Frequency ({unit})")
    plt.title("Rabi frequency vs {parameterToChange}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    figPath = os.path.join(DM.figuresDir, f"rabi_freq_vs_{parameterToChange}.png")
    plt.savefig(figPath)
    plt.close()

    periodsNs = np.array(rabiPeriods)[:,1]
    plt.figure(figsize=(8,5))
    plt.plot(rabiPeriods[:,0], periodsNs, "o-", label="Rabi period")
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
    plt.plot(symmetryAxes[:,0], symmetryAxes[:,1], "o-", label="Detuning at interaction maxima")
    plt.xlabel(f"{parameterToChange}")
    plt.ylabel("Detuning (meV)")
    plt.title(f"Detuning at interaction maxima vs {parameterToChange}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    figPath = os.path.join(DM.figuresDir, f"symmetry_axis_vs_{parameterToChange}.png")
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