import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.DynamicsManager import DynamicsManager, DQDParameters, setupLogger
import numpy as np
import matplotlib.pyplot as plt
import logging


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

logging.info("Running detuning protocol...")

expectedPeriod = 1.4862 # Expected period in ns
interactionTime = expectedPeriod*5.5 # Interaction time in ns for (add half periods to transfer population at its maximum)
intervalTimes = [5*expectedPeriod, interactionTime, 5*expectedPeriod, 5*expectedPeriod] # First solpe, anticrossingn plateau, second slope, final plateau in ns
totalPoints = 600
runOptions = DM.getRunOptions(atol=1e-8, rtol=1e-6, nsteps=10000)
T1 = 100000  # Spin relaxation time in ns
T2star = 100000  # Dephasing time in ns
activateDephasing = False
activateSpinRelaxation = False
cutOffN = None
filter = False

inverseProtocol = True  # If True, the protocol is inverted (from high to low detuning)


spinRelaxation = None
dephasing = None
if activateSpinRelaxation:
    spinRelaxation = DM.gammaFromTime(T1)  # Spin relaxation time in meV
if activateDephasing:
    dephasing = DM.gammaFromTime(T2star)  # Dephasing time in meV

DM.fixedParameters["DecoherenceTime"] = DM.decoherenceTime(T2star, T1)
    
    
if inverseProtocol:
    tlistNano, eiValues = DM.obtainInverseProtocolParameters(intervalTimes, totalPoints, interactionDetuning=interactionDetuning)
else:
    tlistNano, eiValues = DM.obtainOriginalProtocolParameters(intervalTimes, totalPoints, interactionDetuning=interactionDetuning)

populations = DM.detuningProtocol(tlistNano, eiValues, filter=filter, dephasing=dephasing, spinRelaxation=spinRelaxation, cutOffN=cutOffN, runOptions=runOptions)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(10, 10), height_ratios=[3, 1, 1])

title = "Population dynamics"

if cutOffN is not None:
    title += f" (cutOff for {cutOffN} states)"
else:
    title += " (SWT)"

if activateDephasing:
    if not activateSpinRelaxation:
        title += f" with T2 {T2star:.3e} ns"
    else:
        title += f" with T1 {T1:.3e} ns and T2 {T2star:.3e} ns"

else:
    if activateSpinRelaxation:
        title += f" with T1 {T1:.3e} ns"

# Individual populations
statesToPlot = [DM.correspondence[i] for i in range(4)]
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
ax3.plot(tlistNano, sum5States, label='Total (4 states)', linestyle='-', color='tab:red')
ax3.set_xlabel('Time (ns)')
ax3.set_ylabel('Populations')
ax3.set_title('Total populations in relevant subspace')
ax3.legend()
ax3.grid()

plt.subplots_adjust(hspace=0.4, right=0.75)

DM.saveResults(name="Detuning_protocol")
logging.info("All computations ended.")

plt.show()