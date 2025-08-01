from src.PlotsManager import PlotsManager, PlotParameters, BasisToProject, DQDParameters, TypeOfPlot
import numpy as np


# In this script we are interested in see how each eigenstate is similar to any of the basis states of a given basis
# We have first to define the parameters of the BLG DQD we want to simulate

gOrtho = 10
U0 = 8.5
U1 = 0.1
Ei = 0.0
fixedParameters = {
    DQDParameters.B_FIELD.value: 0.2,
    DQDParameters.B_PARALLEL.value: 0.15,
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
    PlotParameters.TYPE: TypeOfPlot.PROJECTION,
    PlotParameters.NUMBER_OF_EIGENSTATES: 22,
    PlotParameters.FIXED_PARAMETERS: fixedParameters,
    PlotParameters.SHOW : True,
    PlotParameters.EXTRA_FOLDER: "",
    PlotParameters.SPECIAL_TITLE: None,
    PlotParameters.PARAM_TO_ITER: DQDParameters.E_I,
    PlotParameters.ARRAY: np.linspace(6.0, 12.0, 1000),
    PlotParameters.BASIS: BasisToProject.SINGLET_TRIPLET_REORDERED_BASIS,
    PlotParameters.Y_LIMS: None
}

# We now initialize the PlotsManager and execute the plotting

pm = PlotsManager(plottingOptions)
pm.plotSimulation()