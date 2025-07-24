from plotting_methods import plot_energy_levels, DQDParameters, ScatterColorOptions, load_and_plot_from_npz, BasisToProject, plot_energy_levels_in_projected_subspace
import numpy as np

# If the data is already computed and stored one can just load it

filepath = "C:\\Users\\Montes\\Documents\\GitHub\\BLG_methods\\DQD_2particles_1orbital\\data\\energy_data_2025-07-22_11-23-30.npz"
load_data = False # Set to False to create a new DQD using the parameters below


# Set global options
simple_initial_states = False # Set to true if the initial eigenstates (for 0 detuning) match approximately the determinant basis states (Rohling 20212)

gOrtho = 1
U0 = 8.5
number_of_eigenstates = 28
parameter_to_change = DQDParameters.E_I # do not use value
array_of_values = np.linspace(0.0, 12.0, 1000)

coloring_option = ScatterColorOptions.ORBITAL_SYMMETRY # ScatterColorOptions.SPIN_SYMMETRY

# If one wnat to project onto a subspace

project = False
basis_to_project = BasisToProject.SINGLET_TRIPLET_BASIS



# Example usage
fixedParameters = {
        DQDParameters.B_FIELD.value: 0.75,  # Set B-field
        DQDParameters.B_PARALLEL.value: 0.0,  # Set parallel magnetic field to zero
        DQDParameters.E_I.value: 0.0,  # Set detuning to zero
        DQDParameters.T.value: 0.04,  # Set hopping parameter
        DQDParameters.DELTA_SO.value: 0.06,  # Set Kane
        DQDParameters.DELTA_KK.value: 0.02,  # Set valley mixing
        DQDParameters.T_SOC.value: 0.01,  # Set spin-flip
        DQDParameters.U0.value: U0,  # Set on-site Coul
        DQDParameters.U1.value: 0.1,  # Set nearest-neighbour Coulomb potential
        DQDParameters.X.value: 0.02,  # Set intersite exchange interaction
        DQDParameters.G_ORTHO.value: gOrtho,  # Set orthogonal component of tunneling corrections
        DQDParameters.G_ZZ.value: 0.1*gOrtho,  # Set correction along
        DQDParameters.G_Z0.value: 2*gOrtho/3,  # Set correction for Z-O mixing
        DQDParameters.G_0Z.value: 2*gOrtho/3,  # Set correction for O-Z mixing
        DQDParameters.GS.value: 2.0,  # Set spin g-factor
        DQDParameters.GV.value: 20.0,  # Set valley g-factor
        DQDParameters.A.value: 0.1,  # Set density assisted hopping
        DQDParameters.P.value: 0.02,  # Set pair hopping interaction
        DQDParameters.J.value: 0.075/gOrtho,  # Set renormalization parameter for Coulomb corrections
}


if load_data:
    print("Loading precomputed data")
    load_and_plot_from_npz(filepath, simpleInitialStates=simple_initial_states)

else:
    if not project:
        plot_energy_levels(number_of_eigenstates, fixedParameters, parameter_to_change, array_of_values,coloring_option, simple_initial_states=simple_initial_states)

    else:
        print(f"Projecting onto {basis_to_project.value} basis")
        plot_energy_levels_in_projected_subspace(number_of_eigenstates, fixedParameters, parameter_to_change, array_of_values, basis_to_project)



# Fixed parameters that works for Rohling basis

fixedParameters = {
        DQDParameters.B_FIELD.value: 0.5,  # Set B-field to zero for initial classification
        DQDParameters.B_PARALLEL.value: 0.0,  # Set parallel magnetic field to zero
        DQDParameters.E_I.value: 0.0,  # Set detuning to zero
        DQDParameters.T.value: 0.04,  # Set hopping parameter
        DQDParameters.DELTA_SO.value: 0.00,  # Set Kane Mele
        DQDParameters.DELTA_KK.value: 0.00,  # Set valley mixing
        DQDParameters.T_SOC.value: 0.0,  # Set spin-flip
        DQDParameters.U0.value: U0,  # Set on-site Coul
        DQDParameters.U1.value: 0.001,  # Set nearest-neighbour Coulomb potential
        DQDParameters.X.value: 0.02,  # Set intersite exchange interaction
        DQDParameters.G_ORTHO.value: gOrtho,  # Set orthogonal component of tunneling corrections
        DQDParameters.G_ZZ.value:10*gOrtho,  # Set correction along
        DQDParameters.G_Z0.value: 2*gOrtho/3,  # Set correction for Z-O mixing
        DQDParameters.G_0Z.value: 2*gOrtho/3,  # Set correction for O-Z mixing
        DQDParameters.GS.value: 2,  # Set spin g-factor
        DQDParameters.GSLFACTOR.value: 1.15, 
        DQDParameters.GV.value: 28,  # Set valley g-factor
        DQDParameters.GVLFACTOR.value: 1.05,
        DQDParameters.A.value: 0.1,  # Set density assisted hopping
        DQDParameters.P.value: 0.02,  # Set pair hopping interaction
        DQDParameters.J.value: 0.00075/gOrtho,  # Set renormalization parameter for Coulomb corrections
}

# Fixed Parameters Knothe like

fixedParameters = {
        DQDParameters.B_FIELD.value: 0.5,  # Set B-field
        DQDParameters.B_PARALLEL.value: 0.0,  # Set parallel magnetic field to zero
        DQDParameters.E_I.value: 0.0,  # Set detuning to zero
        DQDParameters.T.value: 0.04,  # Set hopping parameter
        DQDParameters.DELTA_SO.value: 0.06,  # Set Kane
        DQDParameters.DELTA_KK.value: 0.02,  # Set valley mixing
        DQDParameters.T_SOC.value: 0.0,  # Set spin-flip
        DQDParameters.U0.value: U0,  # Set on-site Coul
        DQDParameters.U1.value: 1,  # Set nearest-neighbour Coulomb potential
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