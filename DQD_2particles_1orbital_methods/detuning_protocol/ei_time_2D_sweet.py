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
from datetime import datetime
from matplotlib import cm

# ---------------- logger ----------------
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

# ---------------- dynamics ----------------
def runDynamics(detuning, parameters, times, cutOffN, dephasing, spinRelaxation):
        params = deepcopy(parameters)
        params[DQDParameters.E_I.value] = detuning
        DM = DynamicsManager(params)
        populations =  DM.simpleTimeEvolution(times, cutOffN=cutOffN, dephasing=dephasing, spinRelaxation=spinRelaxation)
        return DM.getCurrent(populations)


# ---------------- freq estimation ----------------
def computeRabiIfCurrent(signal, times, currentThreshold=1e-3, paddingFactor=4):
    maxCurrent = np.max(np.abs(signal))
    if maxCurrent < currentThreshold:
        return np.nan, maxCurrent, np.nan
    else:
        freq, dominance = estimateRabiFrequency(signal, times, paddingFactor)
        return freq, maxCurrent, dominance
    
def estimateRabiFrequency(signal, times, paddingFactor=4):
    """
    Estima la frecuencia dominante de un señal y su dominancia física.
    Dominancia = altura del pico / anchura a mitad de altura (FWHM)
    """
    signal = np.array(signal) - np.mean(signal)
    N = len(signal)
    dt = times[1] - times[0]

    # FFT con padding
    N_padded = paddingFactor * N
    yf = fft(signal, n=N_padded)
    xf = fftfreq(N_padded, dt)[:N_padded // 2]
    spectrum = 2.0 / N * np.abs(yf[:N_padded // 2])
    spectrum[0] = 0.0  # ignorar DC

    # Pico máximo
    peakIdx = np.argmax(spectrum)
    dominantFreq = xf[peakIdx]

    # --- Dominancia física: altura / FWHM ---
    peakHeight = spectrum[peakIdx]
    halfHeight = peakHeight / 2

    # Buscar los índices a izquierda y derecha donde cruza la mitad
    left = peakIdx
    while left > 0 and spectrum[left] > halfHeight:
        left -= 1
    right = peakIdx
    while right < len(spectrum)-1 and spectrum[right] > halfHeight:
        right += 1

    fwhm = (right - left) * (xf[1] - xf[0])  # ancho en GHz
    dominance = peakHeight / fwhm if fwhm > 0 else np.nan

    return dominantFreq, dominance # Dominant frequency in GHz, dominance as ratio



# ---------------- main computation ----------------
def computeRabiFrequencyMap():
    setupLogger()

    gOrtho = 10
    fixedParameters = {
        DQDParameters.B_FIELD.value: 1.50,
        DQDParameters.B_PARALLEL.value: 0.0,
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
        DQDParameters.J.value: 0.00075 / gOrtho,
    }

    maxTime = 7  # ns
    totalPoints = 100  # Reducir para prueba, aumentar para mejor resolución
    totalTimes = 300
    cutOffN = None
    dephasing = None
    spinRelaxation = None

    # Definir rangos para el espacio 2D
    detuningList = np.linspace(4.0, 6.0, totalPoints)
    
    eps = 1e-5
    halfPoints = totalPoints // 2

    left = np.linspace(-1.0, -eps, halfPoints, endpoint=True)
    right = np.linspace(eps, 1.0, totalPoints - halfPoints, endpoint=True)
    bParallelList = np.concatenate((left, right))

    timesNs = np.linspace(0, maxTime, totalTimes)

    maxCores = min(24, cpu_count())
    logging.info(f"Using {maxCores} cores with joblib.")

    # Matrices para almacenar resultados
    rabiFreqMap = np.zeros((len(bParallelList), len(detuningList)))
    dominanceMap = np.zeros((len(bParallelList), len(detuningList)))
    currentMap = np.zeros((len(bParallelList), len(detuningList)))

    # Computar para cada valor de B_parallel
    for i, b_parallel in enumerate(bParallelList):
        logging.info(f"Processing B_parallel = {b_parallel:.4f} ({i+1}/{len(bParallelList)})")
        
        parameters = deepcopy(fixedParameters)
        parameters[DQDParameters.B_PARALLEL.value] = b_parallel
        
        # Computar dinámica para todos los detunings
        currents = Parallel(n_jobs=maxCores)(
            delayed(runDynamics)(detuning, parameters, timesNs, cutOffN, dephasing, spinRelaxation)
            for detuning in detuningList
        )
        currents = np.array(currents)
        
        # Calcular frecuencias de Rabi para cada detuning
        for j, detuning in enumerate(detuningList):
            signal = currents[j, :]
            freq, maxCurrent, dominance = computeRabiIfCurrent(signal, timesNs, currentThreshold=1e-2)
            rabiFreqMap[i, j] = freq
            currentMap[i, j] = maxCurrent
            dominanceMap[i, j] = dominance

    return detuningList, bParallelList, rabiFreqMap, currentMap, dominanceMap, fixedParameters

# ---------------- gradient computation ----------------
def computeGradients(detuningList, bParallelList, rabiFreqMap):
    """
    Calcula gradientes 2D de las frecuencias de Rabi
    """
    # Gradiente respecto a detuning (eje x)
    grad_detuning = np.gradient(rabiFreqMap, detuningList, axis=1)
    
    # Gradiente respecto a B_parallel (eje y)
    grad_bparallel = np.gradient(rabiFreqMap, bParallelList, axis=0)
    
    # Magnitud del gradiente total
    grad_magnitude = np.sqrt(grad_detuning**2 + grad_bparallel**2)
    
    return grad_detuning, grad_bparallel, grad_magnitude

# ---------------- sweet spots detection ----------------
def findSweetSpots2D(grad_magnitude, threshold_factor=0.05):
    """
    Encuentra sweet spots donde la magnitud del gradiente es mínima
    """
    min_grad = np.min(grad_magnitude)
    max_grad = np.max(grad_magnitude)
    threshold = min_grad + threshold_factor * (max_grad - min_grad)
    
    sweet_spots = grad_magnitude <= threshold
    return sweet_spots

# ---------------- plotting ----------------
def plotResults(detuningList, bParallelList, rabiFreqMap, grad_detuning, dominanceMap):
    """
    Crea gráficas 2D de los resultados.
    Las regiones descartadas (NaN en rabiFreqMap) se muestran en gris claro
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Selected indices of specific b_values
    selected_indices = [
    np.argmin(np.abs(bParallelList - (-0.85))),
    np.argmin(np.abs(bParallelList - (-0.25))),
    np.argmin(np.abs(bParallelList - (0.15))),
    np.argmin(np.abs(bParallelList - (0.75)))
    ]
    colors = ['red', 'blue', 'green', 'purple']
    
    # Crear colormap modificado para mostrar NaN en gris
    cmap_rabi = cm.get_cmap('viridis_r').copy()
    cmap_rabi.set_bad(color='lightgray')
    
    # 1. Mapa de frecuencias de Rabi
    im1 = axes[0, 0].imshow(rabiFreqMap, aspect='auto', 
                           extent=[detuningList[0], detuningList[-1], 
                                  bParallelList[0], bParallelList[-1]],
                           origin='lower', cmap=cmap_rabi)
    axes[0, 0].set_xlabel('Detuning (meV)')
    axes[0, 0].set_ylabel('B_parallel (T)')
    axes[0, 0].set_title('Rabi Frequency Map (GHz)')
    plt.colorbar(im1, ax=axes[0, 0], label="GHz")

    for idx, color in zip(selected_indices, colors):
        axes[0, 0].axhline(y=bParallelList[idx], color=color, linestyle='--', alpha=0.7, linewidth=1.5)
    
    # 2. Gradiente con signo respecto a detuning
    im2 = axes[0, 1].imshow(grad_detuning, aspect='auto',
                        extent=[detuningList[0], detuningList[-1],
                                bParallelList[0], bParallelList[-1]],
                        origin='lower', cmap='bwr', vmin=-np.max(np.abs(grad_detuning)), vmax=np.max(np.abs(grad_detuning)))
    axes[0, 1].set_xlabel('Detuning (meV)')
    axes[0, 1].set_ylabel('B_parallel (T)')
    axes[0, 1].set_title('Gradient wrt Detuning (GHz per meV)')
    plt.colorbar(im2, ax=axes[0, 1], label="d(freq)/d(detuning)")

    for idx, color in zip(selected_indices, colors):
        axes[0, 1].axhline(y=bParallelList[idx], color=color, linestyle='--', alpha=0.7, linewidth=1.5)
    
    # 3. Mapa de dominancia de la frecuencia
    im3 = axes[1, 0].imshow(dominanceMap, aspect='auto',
                        extent=[detuningList[0], detuningList[-1],
                                bParallelList[0], bParallelList[-1]],
                        origin='lower', cmap='inferno')
    axes[1, 0].set_xlabel('Detuning (meV)')
    axes[1, 0].set_ylabel('B_parallel (T)')
    axes[1, 0].set_title('Dominance: Peak Height / FWHM')
    plt.colorbar(im3, ax=axes[1, 0], label="Dominance (Height/GHz)")

    for idx, color in zip(selected_indices, colors):
        axes[1, 0].axhline(y=bParallelList[idx], color=color, linestyle='--', alpha=0.7, linewidth=1.5)
    
    # 4. Cortes transversales (solo frecuencia)
    ax4 = axes[1, 1]

    for idx, color in zip(selected_indices, colors):
        b_val = bParallelList[idx]
        rabiRow = rabiFreqMap[idx, :]

        # Frecuencia (eje izquierdo)
        ax4.plot(detuningList, rabiRow, color=color, label=f'B_parallel = {b_val:.3f} T')

    # Configuración de ejes y leyendas
    ax4.set_xlabel('Detuning (meV)')
    ax4.set_ylabel('Frequency (GHz)')
    ax4.set_title('Transverse Cuts')
    ax4.legend(loc='upper right')
    ax4.grid(True)
    
    plt.tight_layout()
    return fig, axes



if __name__ == "__main__":
    # Computar el mapa 2D
    detuningList, bParallelList, rabiFreqMap, currentMap, dominanceMap, fixedParameters = computeRabiFrequencyMap()
    
    # Calcular gradientes
    grad_detuning, grad_bparallel, grad_magnitude = computeGradients(detuningList, bParallelList, rabiFreqMap)
    
    # Graficar resultados
    fig, axes = plotResults(detuningList, bParallelList, rabiFreqMap, grad_detuning, dominanceMap)
    
    # Guardar datos para análisis posterior
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = os.path.join(os.getcwd(), "DQD_2particles_1orbital_methods", "figures")
    os.makedirs(directory, exist_ok=True)
    plt.savefig(os.path.join(directory, f"rabi_frequency_analysis_{timestamp}.png"))

    np.savez(os.path.join(directory, f"rabi_frequency_analysis_data_{timestamp}.npz"),
             detuningList=detuningList,
             bParallelList=bParallelList,
             rabiFreqMap=rabiFreqMap,
             grad_magnitude=grad_magnitude,
             grad_detuning=grad_detuning,
             currentMap=currentMap,
             fixedParameters=fixedParameters,
             dominanceMap=dominanceMap)
    
    paramPath = os.path.join(directory, f"parameters_{timestamp}.txt")
    with open(paramPath, 'w') as f:
        for key, value in fixedParameters.items():
            if key in [DQDParameters.B_PARALLEL.value, DQDParameters.E_I.value]:
                continue
            f.write(f"{key}: {value}\n")

    print(f"Parameters saved at: {paramPath}")
    
    logging.info("Analysis completed and results saved.")