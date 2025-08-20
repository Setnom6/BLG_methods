import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.DynamicsManager import DynamicsManager, DQDParameters
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

gOrtho = 10
fixedParameters = {
                DQDParameters.B_FIELD.value: 1.50,
                DQDParameters.B_PARALLEL.value: 1.35,
                DQDParameters.E_I.value: 5.5726,
                DQDParameters.T.value: 0.0527,
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

DM = DynamicsManager(fixedParameters)
maxTime = 2.5 # ns
totalPoints = 600
times = np.linspace(0, maxTime, totalPoints)

cutOffN = None
dephasing = None
spinRelaxation = None

initialState = None # for ground state keep it in None

populations = DM.simpleTimeEvolution(times, initialState, cutOffN=cutOffN, dephasing=dephasing, spinRelaxation=spinRelaxation)
currents = DM.getCurrent(populations)

# Obtain frequency
I_t_centered = currents - np.mean(currents)
N = len(I_t_centered)
T = times[1] - times[0]

paddingFactor = 4
N_padded = paddingFactor * N
yf = fft(I_t_centered, n=N_padded)
xf = fftfreq(N_padded, T)[:N_padded // 2]
spectrum = 2.0 / N * np.abs(yf[:N_padded // 2])

i = np.argmax(spectrum)
if 1 <= i < len(spectrum) - 1:
    y0, y1, y2 = spectrum[i - 1], spectrum[i], spectrum[i + 1]
    dx = (y2 - y0) / (2 * (2 * y1 - y2 - y0))
    dominantFrequency = xf[i] + dx * (xf[1] - xf[0])
else:
    dominantFrequency = xf[i]
periodNs = 1.0 / dominantFrequency if dominantFrequency > 0 else np.inf

print(f"Dominant oscillation frequency: {dominantFrequency:.3f} GHz")
print(f"Dominant oscillation period: {periodNs:.3f} ns")


# Plot currents

fig = plt.figure(figsize=(8, 5))
plt.plot(times, currents, lw=2)
plt.xlabel("Time (ns)")
plt.ylabel("I (no Pauli Blockade)")

title = "Current vs time"
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

title += f" (Ei = {fixedParameters[DQDParameters.E_I.value]:.3f} meV)"

plt.title(title)
plt.grid(True)

DM.saveResults(name="Rabi_oscillations_1D")

plt.show()
