import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.PlotsManager import PlotsManager, PlotParameters, BasisToProject, DQDParameters, TypeOfPlot


# In this script we are interested in see how the matrix elements of a hamiltonian in a given basis are conected
# We have first to define the parameters of the BLG DQD we want to simulate

eiValues = np.linspace(0.0, 15.5, 30)
for Ei in eiValues:
    gOrtho = 10
    U0 = 8.5
    U1 = 0.1
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

    # Then, we define the particular options for the plotting


    plottingOptions = {
        PlotParameters.TYPE: TypeOfPlot.HEATMAP,
        PlotParameters.NUMBER_OF_EIGENSTATES: 28,
        PlotParameters.FIXED_PARAMETERS: fixedParameters,
        PlotParameters.SHOW : False,
        PlotParameters.EXTRA_FOLDER: "forVideo",
        PlotParameters.SPECIAL_TITLE: None,
        PlotParameters.BASIS: BasisToProject.SINGLET_TRIPLET_REORDERED_BASIS,
        PlotParameters.BLOCKS: None, # Predeterminated blocks works fine
    }

    # We now initialize the PlotsManager and execute the plotting

    pm = PlotsManager(plottingOptions)
    pm.plotSimulation()
    time.sleep(1)
    plt.close()







