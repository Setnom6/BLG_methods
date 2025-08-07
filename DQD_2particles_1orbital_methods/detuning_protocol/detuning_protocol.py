from DynamicsManager import DynamicsManager, DQDParameters
import numpy as np
import matplotlib.pyplot as plt


gOrtho = 10
U0 = 8.5
U1 = 0.1
Ei = 8.25
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

intervalTimes = [0.6, 0.5, 0.05, 1.0] # First solpe, anticrossingn plateau, second slope, final plateau in ns
totalPoints = 300

populations, tlistNano, eiValues = DM.detuningProtocol(intervalTimes, totalPoints)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(10, 10), height_ratios=[3, 1, 1])

# Individual populations
statesToPlot = [DM.correspondence[i] for i in range(5)]
for label in statesToPlot:
    index = DM.invCorrespondence[label]
    ax1.plot(tlistNano, populations[:, index], label=label)
ax1.set_ylabel('Population')
ax1.set_title('Population dynamics (SWT with detuning sweep)')
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