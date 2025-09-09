import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.PlotsManager import PlotsManager, PlotParameters, BasisToProject, DQDParameters, TypeOfPlot
import numpy as np


# In this script we are interested in see how each eigenstate is similar to any of the basis states of a given basis
# We have first to define the parameters of the BLG DQD we want to simulate

gOrtho = 10
interactionDetuning = 4.7638  # Interaction detuning in meV
fixedParameters = {
            DQDParameters.B_FIELD.value: 1.50,
            DQDParameters.B_PARALLEL.value: 0.5,
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

# Then, we define the particular options for the plotting


plottingOptions = {
    PlotParameters.TYPE: TypeOfPlot.PROJECTION,
    PlotParameters.NUMBER_OF_EIGENSTATES: 22,
    PlotParameters.FIXED_PARAMETERS: fixedParameters,
    PlotParameters.SHOW : True,
    PlotParameters.EXTRA_FOLDER: "",
    PlotParameters.SPECIAL_TITLE: None,
    PlotParameters.PARAM_TO_ITER: DQDParameters.E_I,
    PlotParameters.ARRAY: np.linspace(0.0, 1.5*6.0, 1000),
    PlotParameters.BASIS: BasisToProject.SINGLET_TRIPLET_REORDERED_BASIS,
    PlotParameters.Y_LIMS: None
}

# We now initialize the PlotsManager and execute the plotting

pm = PlotsManager(plottingOptions)
pm.plotSimulation()