from DynamicsManager import DynamicsManager, DQDParameters
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

gOrtho = 10
U0 = 8.5
U1 = 0.1
Ei = 2*8.25
bx = 0.179
fixedParameters = {
            DQDParameters.B_FIELD.value: 0.20,
            DQDParameters.B_PARALLEL.value: bx,
            DQDParameters.E_I.value: Ei,
            DQDParameters.T.value: 0.004,
            DQDParameters.DELTA_SO.value: 0.06,
            DQDParameters.DELTA_KK.value: 0.02,
            DQDParameters.T_SOC.value: 0.0,
            DQDParameters.U0.value: U0,
            DQDParameters.U1.value: U1,
            DQDParameters.X.value: 0.02,
            DQDParameters.G_ORTHO.value: gOrtho,
            DQDParameters.G_ZZ.value: 10 * gOrtho,
            DQDParameters.G_Z0.value: 2 * gOrtho / 3,
            DQDParameters.G_0Z.value: 2 * gOrtho / 3,
            DQDParameters.GS.value: 2,
            DQDParameters.GSLFACTOR.value: 1.0,
            DQDParameters.GV.value: 20.0,
            DQDParameters.GVLFACTOR.value: 0.66,
            DQDParameters.A.value: 0.1,
            DQDParameters.P.value: 0.02,
            DQDParameters.J.value: 0.00075 / gOrtho,
}

DM = DynamicsManager(fixedParameters)
maxTime = 2.5 # ns
totalPoints = 300
times = np.linspace(0, maxTime, totalPoints)

initialState = None # for ground state keep it in None
cutOffN = 5 # This time we will just get the first N states withouth SWT

populations = DM.simpleTimeEvolution(times, initialState, cutOffN=cutOffN)
currents = DM.getCurrent(populations)

# Obtain frequency
I_t_centered = currents - np.mean(currents)
N = len(currents)
T = times[1] - times[0]

yf = fft(I_t_centered)
xf = fftfreq(N, T)[:N // 2] 
spectrum = 2.0 / N * np.abs(yf[0:N // 2])

dominantIndex = np.argmax(spectrum)
dominantFrequency = xf[dominantIndex]  # in GHz (as T is in ns)
periodNs = 1.0 / dominantFrequency if dominantFrequency > 0 else np.inf

print(f"Dominant oscillation frequency: {dominantFrequency:.3f} GHz")
print(f"Dominant oscillation period: {periodNs:.3f} ns")


# Plot currents

fig = plt.figure(figsize=(8, 5))
plt.plot(times, currents, lw=2)
plt.xlabel("Time (ns)")
plt.ylabel("I (no Pauli Blockade)")
plt.title(f"Current vs time (Ei = {fixedParameters[DQDParameters.E_I.value]:.3f} meV,"+
               f" bx = {fixedParameters[DQDParameters.B_PARALLEL.value]:.3f} T," + 
              f"  bx = {fixedParameters[DQDParameters.B_FIELD.value]:.3f} T)")
plt.grid(True)

DM.saveResults(name=f"Rabi_oscillations_1D_cutOff{N}")

plt.show()