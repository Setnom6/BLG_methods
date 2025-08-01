import numpy as np
from src.DQD_2particles_1orbital import DQDParameters
import matplotlib.pyplot as plt

# Parameters

# Single-particle eigenenrgies
gOrtho = 10
U0 = 8.5
U1 = 0.1

values =  np.linspace(0.0, 1.5*U0)
LRTMinusTMinus = np.zeros_like(values)
LRTPlusTMinus = np.zeros_like(values)
LLTMinusS = np.zeros_like(values)
LLTPlusTS = np.zeros_like(values)

for i, Ei in enumerate(values):
    parameters = {
            DQDParameters.B_FIELD.value: 0.05,  # Set B-field to zero for initial classification
            DQDParameters.B_PARALLEL.value: 0.0,  # Set parallel magnetic field to zero
            DQDParameters.E_I.value: Ei,  # Set detuning to zero
            DQDParameters.T.value: 0.004,  # Set hopping parameter
            DQDParameters.DELTA_SO.value: 0.06,  # Set Kane Mele
            DQDParameters.DELTA_KK.value: 0.02,  # Set valley mixing
            DQDParameters.T_SOC.value: 0.0,  # Set spin-flip
            DQDParameters.U0.value: U0,  # Set on-site Coul
            DQDParameters.U1.value: U1,  # Set nearest-neighbour Coulomb potential
            DQDParameters.X.value: 0.02,  # Set intersite exchange interaction
            DQDParameters.G_ORTHO.value: gOrtho,  # Set orthogonal component of tunneling corrections
            DQDParameters.G_ZZ.value: 10*gOrtho,  # Set correction along
            DQDParameters.G_Z0.value: 2*gOrtho/3,  # Set correction for Z-O mixing
            DQDParameters.G_0Z.value: 2*gOrtho/3,  # Set correction for O-Z mixing
            DQDParameters.GS.value: 2,  # Set spin g-factor
            DQDParameters.GSLFACTOR.value: 1.0, 
            DQDParameters.GV.value: 20.0,  # Set valley g-factor
            DQDParameters.GVLFACTOR.value: 1.0,
            DQDParameters.A.value: 0.1,  # Set density assisted hopping
            DQDParameters.P.value: 0.02,  # Set pair hopping interaction
            DQDParameters.J.value: 0.00075/gOrtho,  # Set renormalization parameter for Coulomb corrections
            DQDParameters.MUB.value: 0.05788
    }


    # Computations
    mub = parameters[DQDParameters.MUB.value]
    b_field = parameters[DQDParameters.B_FIELD.value]
    deltaSO = parameters[DQDParameters.DELTA_SO.value]

    # Left dot

    EL = 0.0
    gsL = parameters[DQDParameters.GS.value] * parameters[DQDParameters.GSLFACTOR.value]
    gvL = parameters[DQDParameters.GV.value] * parameters[DQDParameters.GVLFACTOR.value]

    LUpPlus = EL + 0.5*mub * (gvL*b_field + gsL*b_field) + 0.5*deltaSO
    LDownPlus = EL + 0.5*mub * (gvL*b_field - gsL*b_field) - 0.5*deltaSO
    LUpMinus = EL + 0.5*mub * (-gvL*b_field + gsL*b_field) - 0.5*deltaSO
    LDownMinus = EL + 0.5*mub * (-gvL*b_field - gsL*b_field) + 0.5*deltaSO

    """print("LDown-:", LDownMinus)
    print("LUp-:", LUpMinus)
    print("LDown+:", LDownPlus)
    print("LUp+", LUpPlus)
    print("\n")"""

    # Right dot

    ER = parameters[DQDParameters.E_I.value]
    gsR = parameters[DQDParameters.GS.value] 
    gvR = parameters[DQDParameters.GV.value]

    RUpPlus = ER + 0.5*mub * (gvR*b_field + gsR*b_field) + 0.5*deltaSO
    RDownPlus = ER + 0.5*mub * (gvR*b_field - gsR*b_field) - 0.5*deltaSO
    RUpMinus = ER + 0.5*mub * (-gvR*b_field + gsR*b_field) - 0.5*deltaSO
    RDownMinus = ER + 0.5*mub * (-gvR*b_field - gsR*b_field) + 0.5*deltaSO

    """print("RDown-:", RDownMinus)
    print("RUp-:", RUpMinus)
    print("RDown+:", RDownPlus)
    print("RUp+", RUpPlus)
    print("\n")"""


    """print("LR,T-,T-", LDownMinus+RDownMinus)
    print("LR,T0, T-", 0.5*(LDownMinus+RUpMinus+LUpMinus+RDownMinus))
    print("LR,T+,T-", LUpMinus+RUpMinus)"""

    LRTMinusTMinus[i] = LDownMinus+RDownMinus -ER + U1 # We substract the detuning ER
    LRTPlusTMinus[i] = LUpMinus+RUpMinus -ER + U1
    LLTMinusS[i] = LDownMinus+LDownPlus -ER + U0
    LLTPlusTS[i] = LUpMinus+LUpPlus -ER + U0

plt.figure(figsize=(10, 6))

plt.plot(values, LRTMinusTMinus, label='LR, T-, T-', linestyle='-', color='blue')
plt.plot(values, LRTPlusTMinus, label='LR, T+, T-', linestyle='--', color='red')
plt.plot(values, LLTMinusS, label='LL, T-, S', linestyle='-.', color='green')
plt.plot(values, LLTPlusTS, label='LL, T+, S', linestyle=':', color='purple')

plt.xlabel('Ei (meV)')
plt.ylabel('Energy (meV)')
plt.title('Energy levels vs Detuning')
plt.legend()
plt.grid(True)






def findCurveCrossings(xValues, curve1, curve2):
    crossings = []
    diff = curve1 - curve2
    signs = np.sign(diff)

    for i in range(len(signs) - 1):
        if signs[i] != signs[i+1]:
            x0, x1 = xValues[i], xValues[i+1]
            y0, y1 = diff[i], diff[i+1]
            xCross = x0 - y0 * (x1 - x0) / (y1 - y0)
            yCross = np.interp(xCross, xValues, curve1)
            crossings.append((xCross, yCross))
    
    return crossings

crossings1 = findCurveCrossings(values, LRTMinusTMinus, LLTMinusS)
crossings2 = findCurveCrossings(values, LRTPlusTMinus, LLTPlusTS)

print("Crossing between LR T-,T- y LL T-,S:")
for xc, yc in crossings1:
    print(f"Ei = {xc:.3f} meV, Energy = {yc:.3f} meV")

print("\nCrossing between LR T+,T- y LL T+,S:")
for xc, yc in crossings2:
    print(f"Ei = {xc:.3f} meV, Energy = {yc:.3f} meV")


plt.show()