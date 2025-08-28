import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.PlotsManager import PlotsManager, PlotParameters, BasisToProject, DQDParameters, TypeOfPlot
import numpy as np


# In this script we are interested in see how each eigenstate is similar to any of the basis states of a given basis
# We have first to define the parameters of the BLG DQD we want to simulate

tValues = [0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3]
bValues = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
deltaSOValues = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
U1Values = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
deltaKKVlues = [-0.1, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
for deltaKK in deltaKKVlues:
    gOrtho = 10
    interactionDetuning = 4.7638  # Interaction detuning in meV
    fixedParameters = {
            DQDParameters.B_FIELD.value: 1.5,
            DQDParameters.B_PARALLEL.value: 0.1,
            DQDParameters.E_I.value: interactionDetuning,
            DQDParameters.T.value: 0.05,
            DQDParameters.DELTA_SO.value: 0.066,
            DQDParameters.DELTA_KK.value: deltaKK,
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


    # Then, we define the particular options for the plotting


    plottingOptions = {
        PlotParameters.TYPE: TypeOfPlot.PROJECTION,
        PlotParameters.NUMBER_OF_EIGENSTATES: 22,
        PlotParameters.FIXED_PARAMETERS: fixedParameters,
        PlotParameters.SHOW : False,
        PlotParameters.EXTRA_FOLDER: "forVideo",
        PlotParameters.SPECIAL_TITLE: f"DeltaKK' = {deltaKK:.3f}",
        PlotParameters.PARAM_TO_ITER: DQDParameters.E_I,
        PlotParameters.ARRAY: np.linspace(0.0, 1.5*6.0, 1000),
        PlotParameters.BASIS: BasisToProject.SINGLET_TRIPLET_REORDERED_BASIS,
        PlotParameters.Y_LIMS: [-1.5, 0.5]
    }

    # We now initialize the PlotsManager and execute the plotting

    pm = PlotsManager(plottingOptions)
    pm.plotSimulation()







