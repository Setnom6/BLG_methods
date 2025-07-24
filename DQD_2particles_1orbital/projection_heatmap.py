import matplotlib.pyplot as plt
import numpy as np
from plotting_methods import hamiltonian_heatmap, DQDParameters, BasisToProject
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


# Parameters
gOrtho = 1
U0 = 8.5
fixedParameters = {
        DQDParameters.B_FIELD.value: 0.5,  # Set B-field
        DQDParameters.B_PARALLEL.value: 0.0,  # Set parallel magnetic field to zero
        DQDParameters.E_I.value: 1.0*U0,  # Set detuning
        DQDParameters.T.value: 0.04,  # Set hopping parameter
        DQDParameters.DELTA_SO.value: 0.06,  # Set Kane
        DQDParameters.DELTA_KK.value: 0.02,  # Set valley mixing
        DQDParameters.T_SOC.value: 0.0,  # Set spin-flip
        DQDParameters.U0.value: U0,  # Set on-site Coul
        DQDParameters.U1.value: 0.01,  # Set nearest-neighbour Coulomb potential
        DQDParameters.X.value: 0.02,  # Set intersite exchange interaction
        DQDParameters.G_ORTHO.value: gOrtho,  # Set orthogonal component of tunneling corrections
        DQDParameters.G_ZZ.value: 10*gOrtho,  # Set correction along
        DQDParameters.G_Z0.value: 2*gOrtho/3,  # Set correction for Z-O mixing
        DQDParameters.G_0Z.value: 2*gOrtho/3,  # Set correction for O-Z mixing
        DQDParameters.GS.value: 2.0,  # Set spin g-factor
        DQDParameters.GV.value: 28.0,  # Set valley g-factor
        DQDParameters.A.value: 0.1,  # Set density assisted hopping
        DQDParameters.P.value: 0.02,  # Set pair hopping interaction
        DQDParameters.J.value: 0.075/gOrtho,  # Set renormalization parameter for Coulomb corrections
}

hamiltonian_heatmap(fixedParameters, BasisToProject.SINGLET_TRIPLET_BASIS, compare_with_original=False)
