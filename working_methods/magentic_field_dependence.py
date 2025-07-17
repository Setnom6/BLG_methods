from Knothe2024 import Knothe2024, DQDParameters
import numpy as np
import matplotlib.pyplot as plt

def scatter_color_selector(classification: dict):
    """
    We identify symmetric states (S) with -1 value which will be blue
    We identify antisymmetric states (AS) with 1 value which will be red
    """
    most_probably_state = classification['most_similar_state']
    probability_most_likely = classification['probability']

    AS_weights = 0.0
    S_weights = 0.0
    if 'A' in most_probably_state:
        AS_weights += probability_most_likely
    else:
        S_weights += probability_most_likely

    i = 1
    while (AS_weights+S_weights) < 0.85 and i < 16:
        new_state = classification['ordered_probabilities'][i]['label']
        new_prob = classification['ordered_probabilities'][i]['probability']

        if 'A' in new_state:
            AS_weights+=new_prob
        else:
            S_weights+=new_prob
        i+=1


    normalized_AS_weights = AS_weights / (AS_weights+S_weights)
    normalized_S_weights = S_weights / (AS_weights+S_weights)

    return -1*normalized_S_weights + 1*normalized_AS_weights


gOrtho = 90
U0 = 6.0
# Example usage
fixedParameters = {
        DQDParameters.B_FIELD.value: 0.00,  # Set B-field to zero for initial classification
        DQDParameters.B_PARALLEL.value: 0.0,  # Set parallel magnetic field to zero
        DQDParameters.E_I.value: 0.0,  # Set detuning to zero
        DQDParameters.T.value: 0.1,  # Set hopping parameter
        DQDParameters.DELTA_SO.value: 0.068,  # Set Kane
        DQDParameters.DELTA_KK.value: 0.00,  # Set valley mixing
        DQDParameters.T_SOC.value: 0.0,  # Set spin-flip
        DQDParameters.U0.value: U0,  # Set on-site Coul
        DQDParameters.U1.value: U0/4,  # Set nearest-neighbour Coulomb potential
        DQDParameters.X.value: U0/100,  # Set intersite exchange interaction
        DQDParameters.G_ORTHO.value: gOrtho,  # Set orthogonal component of tunneling corrections
        DQDParameters.G_ZZ.value: 4,  # Set correction along
        DQDParameters.G_Z0.value: -gOrtho,  # Set correction for Z-O mixing
        DQDParameters.G_0Z.value: -gOrtho,  # Set correction for O-Z mixing
        DQDParameters.GS.value: 2.0,  # Set spin g-factor
        DQDParameters.GV.value: 20.0,  # Set valley g-factor
        DQDParameters.A.value: 0.0,  # Set density assisted hopping
        DQDParameters.P.value: 0.00,  # Set pair hopping interaction
        DQDParameters.J.value: 4e-4,  # Set renormalization parameter for Coulomb corrections
}

number_of_eigenstates = 16  # Number of eigenstates to plot
arrayToPlot = np.linspace(0.0, 0.4, 1000)  # Example range for B-field or detuning

dqd = Knothe2024()
basis = dqd.FSU.basis
correspondence = dqd.knothe_correspondence
knothe_basis = dqd.knothe_basis

eigvals = np.zeros((len(arrayToPlot), number_of_eigenstates))
colors = np.zeros((len(arrayToPlot), number_of_eigenstates))

for i, bField in enumerate(arrayToPlot):
    parameters_to_change = fixedParameters.copy()
    if bField < 0:
        parameters_to_change[DQDParameters.B_PARALLEL.value] = -bField
    else:
        parameters_to_change[DQDParameters.B_FIELD.value] = bField

    eigval, eigv = dqd.calculate_eigenvalues_and_eigenvectors(parameters_to_change=parameters_to_change)
    
    eigvals[i] = eigval[:number_of_eigenstates].real 

    for j in range(number_of_eigenstates):
        classification = dqd.FSU.classify_eigenstate(knothe_basis, correspondence, eigv[:, j])
        colors[i, j] = scatter_color_selector(classification)

plt.figure(figsize=(10, 6))
for j in range(number_of_eigenstates):
    plt.scatter(arrayToPlot, eigvals[:, j], 
                c=colors[:, j], cmap='bwr', vmin=-1, vmax=1, s=1)
    

plt.xlabel('B field (T)')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues colored by symmetry classification')
plt.colorbar(label='Symmetry: -1 (S) to +1 (AS)')
plt.grid(True)
plt.tight_layout()
plt.show()