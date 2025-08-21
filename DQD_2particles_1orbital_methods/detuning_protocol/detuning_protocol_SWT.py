import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.DynamicsManager import DynamicsManager, DQDParameters
import numpy as np
import matplotlib.pyplot as plt


gOrtho = 10
fixedParameters = {
                DQDParameters.B_FIELD.value: 1.50,
                DQDParameters.B_PARALLEL.value: 0.1586,
                DQDParameters.E_I.value: 3.1839,
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

DM = DynamicsManager(fixedParameters)

intervalTimes = [10, 13.1523, 3, 10] # First solpe, anticrossingn plateau, second slope, final plateau in ns
totalPoints = 1200
runOptions = DM.getRunOptions(atol=1e-8, rtol=1e-6, nsteps=10000)
T1 = 2e5  # Spin relaxation time in ns
T2 = 1e4  # Dephasing time in ns
cutOffN = None
filter = False

dephasing = DM.gammaDephasing(T2, T1)  # Dephasing and spin relaxation time in meV
spinRelaxation = DM.gammaRelaxation(T1)  # Spin relaxation time in meV

tlistNano, eiValues = DM.obtainOriginalProtocolParameters(intervalTimes, totalPoints)
populations = DM.detuningProtocol(tlistNano, eiValues, filter=filter, dephasing=dephasing, spinRelaxation=spinRelaxation, cutOffN=cutOffN, runOptions=runOptions)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(10, 10), height_ratios=[3, 1, 1])

title = "Population dynamics"

if cutOffN is not None:
    title += f" (cutOff for {cutOffN} states)"
else:
    title += " (SWT)"

if dephasing is not None:
    if spinRelaxation is not None:
        title += f" with dephasing {dephasing} and spin relaxation {spinRelaxation}"
    else:
        title += f" with dephasing {dephasing}"
else:
    if spinRelaxation is not None:
        title += f" with spin relaxation {spinRelaxation}"

# Individual populations
statesToPlot = [DM.correspondence[i] for i in range(5)]
for label in statesToPlot:
    index = DM.invCorrespondence[label]
    ax1.plot(tlistNano, populations[:, index], label=label)
ax1.set_ylabel('Population')
ax1.set_title(title)
ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
ax1.grid()

# Detuning E_I(t)
ax2.plot(tlistNano, eiValues, color='black', linewidth=2)
ax2.set_ylabel(r'$E_I$ (meV)')
ax2.set_title('Detuning sweep')
ax2.grid()

# Singlet and triplet subspace populations
sumTriplet = (
        populations[:, DM.invCorrespondence["LR,T+,T-"]] +
        populations[:, DM.invCorrespondence["LR,T0,T-"]] +
        populations[:, DM.invCorrespondence["LR,T-,T-"]]
    )
sumSinglet = (
        populations[:, DM.invCorrespondence["LL,S,T-"]] +
        populations[:, DM.invCorrespondence["LR,S,T-"]]
    )
sum5States = sumTriplet + sumSinglet

ax3.plot(tlistNano, sumTriplet, label='Spatially antisymmetric', linestyle='--', color='tab:blue')
ax3.plot(tlistNano, sumSinglet, label='Spatially symmetric', linestyle='--', color='tab:green')
ax3.plot(tlistNano, sum5States, label='Total (5 states)', linestyle='-', color='tab:red')
ax3.set_xlabel('Time (ns)')
ax3.set_ylabel('Populations')
ax3.set_title('Total populations in relevant subspace')
ax3.legend()
ax3.grid()

plt.subplots_adjust(hspace=0.4, right=0.75)

DM.saveResults(name="Detuning_protocol")

plt.show()