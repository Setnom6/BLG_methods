from src.PlotsManager import PlotsManager, PlotParameters, BasisToProject, DQDParameters, TypeOfPlot
import numpy as np
import time

parameterForVideo = DQDParameters.GVLFACTOR
values = np.linspace(0.05, 2.0, 20)
        # In this script we are interested in see how each eigenstate is similar to any of the basis states of a given basis
        # We have first to define the parameters of the BLG DQD we want to simulate

for i, value in enumerate(values):
        gOrtho = 10
        U0 = 8.5
        fixedParameters = {
                DQDParameters.B_FIELD.value: 0.25,  # Set B-field to zero for initial classification
                DQDParameters.B_PARALLEL.value: 0.0,  # Set parallel magnetic field to zero
                DQDParameters.E_I.value: 0.0,  # Set detuning to zero
                DQDParameters.T.value: 0.004,  # Set hopping parameter
                DQDParameters.DELTA_SO.value: 0.00,  # Set Kane Mele
                DQDParameters.DELTA_KK.value: 0.00,  # Set valley mixing
                DQDParameters.T_SOC.value: 0.0,  # Set spin-flip
                DQDParameters.U0.value: U0,  # Set on-site Coul
                DQDParameters.U1.value: 0.01,  # Set nearest-neighbour Coulomb potential
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

        fixedParameters[parameterForVideo.value] = value
        basis_to_project = BasisToProject.SINGLET_TRIPLET_IN_SPIN_BASIS

        plottingOptions = {
        PlotParameters.TYPE: TypeOfPlot.PROJECTION,
        PlotParameters.NUMBER_OF_EIGENSTATES: 22,
        PlotParameters.FIXED_PARAMETERS: fixedParameters,
        PlotParameters.SHOW : False,
        PlotParameters.EXTRA_FOLDER: "forVideo",
        PlotParameters.SPECIAL_TITLE: f"Eigenvalues in {basis_to_project.value} basis for {parameterForVideo.value} = {value:.3f}",
        PlotParameters.PARAM_TO_ITER: DQDParameters.E_I,
        PlotParameters.ARRAY: np.linspace(6.0, 12.0, 1000),
        PlotParameters.BASIS: basis_to_project,
        PlotParameters.Y_LIMS: [-0.4, 0.4],
        }

        # We now initialize the PlotsManager and execute the plotting

        pm = PlotsManager(plottingOptions)
        pm.plotSimulation()
        print(f"Figure {i+1} of {len(values)} completed")
        time.sleep(0.5)



