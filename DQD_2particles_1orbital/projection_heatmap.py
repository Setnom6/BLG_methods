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
    DQDParameters.B_FIELD.value: 0.5,
    DQDParameters.B_PARALLEL.value: 0.0,
    DQDParameters.E_I.value: 0.3,
    DQDParameters.T.value: 0.04,
    DQDParameters.DELTA_SO.value: 0.06,
    DQDParameters.DELTA_KK.value: 0.02,
    DQDParameters.T_SOC.value: 0.0,
    DQDParameters.U0.value: U0,
    DQDParameters.U1.value: 1,
    DQDParameters.X.value: 0.02,
    DQDParameters.G_ORTHO.value: gOrtho,
    DQDParameters.G_ZZ.value: 10 * gOrtho,
    DQDParameters.G_Z0.value: 2 * gOrtho / 3,
    DQDParameters.G_0Z.value: 2 * gOrtho / 3,
    DQDParameters.GS.value: 2.0,
    DQDParameters.GV.value: 28.0,
    DQDParameters.A.value: 10,
    DQDParameters.P.value:2 ,
    DQDParameters.J.value: 0.075 / gOrtho,
}

hamiltonian_heatmap(fixedParameters, BasisToProject.SPIN_SYMMETRY, compare_with_original=True)
