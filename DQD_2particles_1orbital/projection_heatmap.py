from src.PlotsManager import PlotsManager, PlotParameters, BasisToProject, DQDParameters, TypeOfPlot
import numpy as np


# In this script we are interested in see how the matrix elements of a hamiltonian in a given basis are conected
# We have first to define the parameters of the BLG DQD we want to simulate

gOrtho = 10
U0 = 8.5
U1 = 0.1
fixedParameters = {
            DQDParameters.B_FIELD.value: 0.25,  # Set B-field to zero for initial classification
            DQDParameters.B_PARALLEL.value: 0.2,  # Set parallel magnetic field to zero
            DQDParameters.E_I.value: 0.0,  # Set detuning to zero
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
    }

# Then, we define the particular options for the plotting


plottingOptions = {
    PlotParameters.TYPE: TypeOfPlot.HEATMAP,
    PlotParameters.NUMBER_OF_EIGENSTATES: 28,
    PlotParameters.FIXED_PARAMETERS: fixedParameters,
    PlotParameters.SHOW : True,
    PlotParameters.EXTRA_FOLDER: "",
    PlotParameters.SPECIAL_TITLE: None,
    PlotParameters.BASIS: BasisToProject.SINGLET_TRIPLET_REORDERED_BASIS,
    PlotParameters.BLOCKS: None, # Predeterminated blocks works fine
}

# We now initialize the PlotsManager and execute the plotting

pm = PlotsManager(plottingOptions)
pm.plotSimulation()
