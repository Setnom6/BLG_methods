from src.DQD_2particles_1orbital import DQD_2particles_1orbital, DQDParameters
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from matplotlib.colors import LogNorm

from enum import Enum


class ScatterColorOptions(Enum):
    SINGLET_TRIPLET_BASIS = 'Singlet Triplet Basis symmetry'
    ORBITAL_SYMMETRY = 'Orbital symmetry'
    SPIN_SYMMETRY = 'Spin Symmetry'
    VALLEY_SYMMETRY = 'Valley Symmetry'
    TOTAL_SPIN = 'Total Spin'

class BasisToProject(Enum):
    SINGLET_TRIPLET_BASIS = 'Singlet Triplet Basis symmetry'
    ORBITAL_SYMMETRY = 'Orbital symmetry'
    SPIN_SYMMETRY = 'Spin Symmetry'
    VALLEY_SYMMETRY = 'Valley Symmetry'
    ORIGINAL = 'Original'


def plot_energy_levels(number_of_eigenstates: int, fixed_parameters: dict, parameter_to_iter: DQDParameters, array_to_plot: np.ndarray, scatter_color_option: ScatterColorOptions = None, simple_initial_states = False):
    dqd = DQD_2particles_1orbital(fixed_parameters)

    eigvals = np.zeros((len(array_to_plot), number_of_eigenstates))
    colors = np.zeros((len(array_to_plot), number_of_eigenstates))

    for i, value in enumerate(array_to_plot):
        parameters_to_change = obtain_dict_parameters_to_chnange(fixed_parameters, parameter_to_iter, value)
        eigval, eigv = dqd.calculate_eigenvalues_and_eigenvectors(parameters_to_change=parameters_to_change)
        
        eigvals[i] = eigval[:number_of_eigenstates].real 

        if scatter_color_option is not None:
            for j in range(number_of_eigenstates):
                colors[i, j] = obtain_colors_to_plot(dqd, scatter_color_option, eigv[:, j])

    plt.figure(figsize=(10, 6))
    if parameter_to_iter == DQDParameters.E_I:
        if scatter_color_option is not None:
            for j in range(number_of_eigenstates):
                plt.scatter(array_to_plot, eigvals[:, j] - array_to_plot, c=colors[:, j], cmap='bwr', vmin=-1, vmax=1, s=1)
        else:
            for j in range(number_of_eigenstates):
                plt.scatter(array_to_plot, eigvals[:, j] - array_to_plot, s=1)
    else:
        if scatter_color_option is not None:
            for j in range(number_of_eigenstates):
                plt.scatter(array_to_plot, eigvals[:, j], c=colors[:, j], cmap='bwr', vmin=-1, vmax=1, s=1)
        else:
            for j in range(number_of_eigenstates):
                plt.scatter(array_to_plot, eigvals[:, j], s=1)

    if simple_initial_states:
        for i in range(16):
            plt.plot(array_to_plot[array_to_plot <= 1.0], predicted11state(dqd, array_to_plot[array_to_plot <= 1.0], i))


    labels_dict = obtain_dict_labels(parameter_to_iter, scatter_color_option)
    plt.xlabel(labels_dict["x_axis"])
    plt.ylabel(labels_dict["y axis"])
    plt.title(labels_dict["title"])
    if scatter_color_option is not None:
        plt.colorbar(label=labels_dict["colorbar"])
    plt.grid(True)
    plt.tight_layout()
    save_figure_and_parameters(fixed_parameters)
    save_data_as_npz(fixed_parameters, parameter_to_iter, array_to_plot, eigvals, labels_dict, colors)
    plt.show()




def obtain_dict_parameters_to_chnange(fixed_parameters: dict, parameter_to_iter: DQDParameters, value: float):
    new_dict = fixed_parameters.copy()

    if parameter_to_iter == DQDParameters.B_FIELD:
        if value < 0:
            new_dict[DQDParameters.B_FIELD.value] = 0.0
            new_dict[DQDParameters.B_PARALLEL.value] = -value
        else:
            new_dict[DQDParameters.B_FIELD.value] = value
            new_dict[DQDParameters.B_PARALLEL.value] = 0.0

    else:
        new_dict[parameter_to_iter.value] = value
    
    return new_dict


def obtain_colors_to_plot(dqd: DQD_2particles_1orbital, scatter_color_option: ScatterColorOptions, eigenstate: np.ndarray):
    if scatter_color_option == ScatterColorOptions.ORBITAL_SYMMETRY:
        correspondence = dqd.symmetric_antisymmetric_correspondence
        preferred_basis = dqd.orbital_symmetry_basis
        classification = dqd.FSU.classify_eigenstate(preferred_basis, correspondence, eigenstate)
        return symmetric_antisymmetric_color_difference(classification)
    
    elif scatter_color_option == ScatterColorOptions.SPIN_SYMMETRY:
        correspondence = dqd.symmetric_antisymmetric_correspondence
        preferred_basis = dqd.spin_symmetry_basis
        classification = dqd.FSU.classify_eigenstate(preferred_basis, correspondence, eigenstate)
        return symmetric_antisymmetric_color_difference(classification)
    
    elif scatter_color_option == ScatterColorOptions.VALLEY_SYMMETRY:
        correspondence = dqd.symmetric_antisymmetric_correspondence
        preferred_basis = dqd.valley_symmetry_basis
        classification = dqd.FSU.classify_eigenstate(preferred_basis, correspondence, eigenstate)
        return symmetric_antisymmetric_color_difference(classification)
    
    elif scatter_color_option == ScatterColorOptions.SINGLET_TRIPLET_BASIS:
        return singlet_triplet_symmetry_color(eigenstate)
    
    elif scatter_color_option == ScatterColorOptions.TOTAL_SPIN:
        return total_spin_color(dqd, eigenstate)
    
    else:
        return None



def symmetric_antisymmetric_color_difference(classification: dict):
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

def singlet_triplet_symmetry_color(eigenvector: np.ndarray):
    symmetricMask = np.array([1]*12 + [0]*10 + [1]*6)
    antisymmetricMask = np.array([0]*12 + [1]*10 +[0]*6)

    weightS = np.sum(np.abs(eigenvector)**2 * symmetricMask)
    weightAS = np.sum(np.abs(eigenvector)**2 * antisymmetricMask)
    total = weightS + weightAS

    if total == 0:
        return 0
    return (weightAS - weightS) / total


def total_spin_color(dqd: DQD_2particles_1orbital, state: np.ndarray):
    """
    We identify <S2>=1 (singlet) with value +1 red and <S2>=2 (triplet) with value -1 blue
    """
    total_spin = dqd.obtain_total_spin(state)
    total_spin = total_spin*2-3
    return -total_spin


def obtain_dict_labels(parameter_to_iter: DQDParameters, scatter_color_option: ScatterColorOptions):
    dict_labels = {}
    dict_labels["y axis"] = "Eigvalue" 
    if parameter_to_iter == DQDParameters.E_I:
        dict_labels["x_axis"] = "E_i (meV)"
        dict_labels["y axis"] = "Eigvalue - E_ref (meV)"

    elif parameter_to_iter == DQDParameters.B_FIELD:
        dict_labels["x_axis"] = "B (T)"

    else:
        dict_labels["x_axis"] = parameter_to_iter.value

    dict_labels["title"] = "Energy levels"
    if scatter_color_option is not None:
        dict_labels["colorbar"] = f'{scatter_color_option.value}: blue (S) to red (AS)'
        dict_labels["title"] = "Energy levels colored by symmetry classification"
        if scatter_color_option == ScatterColorOptions.TOTAL_SPIN:
            dict_labels["colorbar"] = f'{scatter_color_option.value}: blue (triplet) to red (singlet)'

    return dict_labels
        

def save_figure_and_parameters(fixed_parameters: dict, figure_title = "energy_plot"):
    figures_dir = os.path.join(os.getcwd(),"DQD_2particles_1orbital", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    fig_path = os.path.join(figures_dir, f"{figure_title}_{timestamp}.svg")
    plt.savefig(fig_path)
    print(f"Figura guardada en: {fig_path}")

    param_path = os.path.join(figures_dir, f"parameters_{timestamp}.txt")
    with open(param_path, 'w') as f:
        for key, value in fixed_parameters.items():
            f.write(f"{key}: {value}\n")
    print(f"Parámetros guardados en: {param_path}")

def predicted11state(dqd: DQD_2particles_1orbital, array, index):
    mub = dqd.parameters[DQDParameters.MUB.value]
    b_field  = dqd.parameters[DQDParameters.B_FIELD.value]
    gsL = dqd.parameters[DQDParameters.GS.value] * dqd.parameters[DQDParameters.GSLFACTOR.value]
    gvL = dqd.parameters[DQDParameters.GV.value] * dqd.parameters[DQDParameters.GVLFACTOR.value]
    gsR = dqd.parameters[DQDParameters.GS.value] 
    gvR = dqd.parameters[DQDParameters.GV.value]
    hsL = 0.5*mub*b_field*gsL
    hvL = 0.5*mub*b_field*gvL
    hsR = 0.5*mub*b_field*gsR
    hvR = 0.5*mub*b_field*gvR

    hs = 0.5*(hsL+hsR)
    hv = 0.5*(hvL+hvR)

    delta_hs = hsL-hsR
    delta_hv = hvL-hvR

    t = dqd.parameters[DQDParameters.T.value]
    U = dqd.parameters[DQDParameters.U0.value]
    J = 4*t**2/U

    if (J > hs*1e-2) or (J > hv*1e-2):
        print("Coupling constant is too big to have simple initial states")

    value = 0.0

    if 0 == index:
        value = -2*(hv+hs)
    elif 1 == index:
        value = -2*hv -delta_hs
    elif 2 == index:
        value = -2*hv+delta_hs
    elif 3 == index:
        value = 2*(hs-hv)
    elif 4 == index:
        value = -2*hs-delta_hv
    elif 5 == index:
        value = -2*hs+delta_hv
    elif 6 == index:
        value = -delta_hv-delta_hs
    elif 7 == index:
        value = delta_hs-delta_hv
    elif 8 == index:
        value = delta_hv- delta_hs
    elif 9 == index:
        value = delta_hv+delta_hs
    elif 10 == index:
        value = 2*hs-delta_hv
    elif 11 == index:
        value = 2*hs + delta_hv
    elif 12 == index:
        value = 2*(hv-hs)
    elif 13 == index:
        value = 2*hv-delta_hs
    elif 14 == index:
        value = 2*hv+delta_hs
    else:
        value = 2*(hv+hs)

    print(value)
    return np.array([value for _ in range(len(array))])

def save_data_as_npz(fixedParameters: dict, parameterToIter: DQDParameters, arrayToPlot: np.ndarray, eigvals: np.ndarray, labelsDict: dict, colors: np.ndarray = None):
    dataDir = os.path.join(os.getcwd(), "DQD_2particles_1orbital", "data")
    os.makedirs(dataDir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    npzPath = os.path.join(dataDir, f"energy_data_{timestamp}.npz")

    np.savez_compressed(
        npzPath,
        fixedParameters=fixedParameters,
        parameterToIter=parameterToIter,
        arrayToPlot=arrayToPlot,
        eigvals=eigvals,
        colors=colors, 
        labelsDict=labelsDict
    )
    print(f"Datos guardados en: {npzPath}")



def load_and_plot_from_npz(npzPath: str, simpleInitialStates: bool = False):
    data = np.load(npzPath, allow_pickle=True)
    fixedParameters = data["fixedParameters"].item()
    parameterToIter = data["parameterToIter"]
    arrayToPlot = data["arrayToPlot"]
    eigvals = data["eigvals"]
    colors = data["colors"] if "colors" in data.files else None
    labelsDict = data["labelsDict"].item()

    numberOfEigenstates = eigvals.shape[1]

    plt.figure(figsize=(10, 6))
    if parameterToIter == DQDParameters.E_I:
        if colors is not None:
            for j in range(numberOfEigenstates):
                plt.scatter(arrayToPlot, eigvals[:, j] - arrayToPlot, c=colors[:, j], cmap='bwr', vmin=-1, vmax=1, s=1)
        else:
            for j in range(numberOfEigenstates):
                plt.scatter(arrayToPlot, eigvals[:, j] - arrayToPlot, s=1)
    else:
        if colors is not None:
            for j in range(numberOfEigenstates):
                plt.scatter(arrayToPlot, eigvals[:, j], c=colors[:, j], cmap='bwr', vmin=-1, vmax=1, s=1)
        else:
            for j in range(numberOfEigenstates):
                plt.scatter(arrayToPlot, eigvals[:, j], s=1)

    if simpleInitialStates:
        dqd = DQD_2particles_1orbital(fixedParameters)
        for i in range(16):
            plt.plot(arrayToPlot[arrayToPlot <= 1.0], predicted11state(dqd, arrayToPlot[arrayToPlot <= 1.0], i))

    plt.xlabel(labelsDict["x_axis"])
    plt.ylabel(labelsDict["y axis"])
    plt.title(labelsDict["title"])
    if colors is not None:
        plt.colorbar(label=labelsDict["colorbar"])
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def hamiltonian_heatmap(fixed_parameters: dict, basis_to_project: BasisToProject, compare_with_original: bool = False):
    dqd = DQD_2particles_1orbital(fixed_parameters)
    if basis_to_project == BasisToProject.SINGLET_TRIPLET_BASIS:
        projectedH = dqd.project_hamiltonian(dqd.singlet_triplet_basis, parameters_to_change=fixed_parameters)
        if compare_with_original:
            originalH = dqd.obtain_hamiltonian_determinant_basis(fixed_parameters)
            absH = np.abs(originalH)-np.abs(projectedH)
            title = "Hamiltonian in original basis - in singlet-triplet basis"
            plot_hamiltonian_no_blocks(absH, title, colormap='RdBu_r')
        else:
            plot_hamiltonian_charge_configuration_blocks(np.abs(projectedH))

    else:
        vectors_list = None
        if basis_to_project == BasisToProject.ORBITAL_SYMMETRY:
            vectors_list = dqd.orbital_symmetry_basis
        elif basis_to_project == BasisToProject.SPIN_SYMMETRY:
            vectors_list = dqd.spin_symmetry_basis
        elif basis_to_project == BasisToProject.VALLEY_SYMMETRY:
            vectors_list = dqd.valley_symmetry_basis
        else:
            pass

        if vectors_list is not None:
            projectedH = dqd.project_hamiltonian(vectors_list, parameters_to_change=fixed_parameters)
            absH = np.abs(projectedH)
        else:
            absH = np.abs(dqd.obtain_hamiltonian_determinant_basis(fixed_parameters))
        plot_hamiltonian_no_blocks(absH, title = f'Hamiltonian in {basis_to_project.value} basis')

    save_figure_and_parameters(fixed_parameters, figure_title='heatmap')
    plt.show()


def plot_hamiltonian_charge_configuration_blocks(absH: np.ndarray, title: str = "Hamiltonian divided by charge configuration", colormap: str = "viridis"):
    """
    Visualizes the projected Hamiltonian by showing the blocks corresponding to each charge sector
    using a logarithmic color scale.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if np.any(absH < 0.0):
        im = ax.imshow(absH, cmap=colormap)

    else:
        # Set a floor value of 1e-5 for the logarithmic scale
        min_val = 1e-5
        absH[absH < min_val] = min_val  # Replace values smaller than 1e-5 with 1e-5
        
        # Create the image with logarithmic scale
        im = ax.imshow(absH, cmap=colormap, norm=LogNorm(vmin=min_val, vmax=np.max(absH)))

    # Add separations for charge sectors
    sectorSizes = {
        '(2,0)': 6,
        '(1,1)': 16,
        '(0,2)': 6
    }

    # Block boundaries
    boundaries = np.cumsum(list(sectorSizes.values()))
    
    for pos in boundaries[:-1]:  # avoid final line out of range
        ax.axhline(pos - 0.5, color='white', linewidth=1.5)
        ax.axvline(pos - 0.5, color='white', linewidth=1.5)

    # Block labels
    middle = lambda start, size: start + size / 2 - 0.5
    starts = np.cumsum([0] + list(sectorSizes.values())[:-1])
    labels = list(sectorSizes.keys())
    
    for i, (start, size, label) in enumerate(zip(starts, sectorSizes.values(), labels)):
        ax.text(middle(start, size), -3, label, ha='center', va='bottom', fontsize=10, color='white', rotation=90)
        ax.text(-3, middle(start, size), label, va='center', ha='right', fontsize=10, color='white')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='log(|HProj|)')
    plt.tight_layout()


def plot_hamiltonian_no_blocks(absH: np.ndarray, title: str = "Hamiltonian", colormap: str = "viridis"):
    """
    Visualizes the projected Hamiltonian.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if np.any(absH < 0.0):
        im = ax.imshow(absH, cmap=colormap)

    else:
        # Set a floor value of 1e-5 for the logarithmic scale
        min_val = 1e-5
        absH[absH < min_val] = min_val  # Replace values smaller than 1e-5 with 1e-5
        
        # Create the image with logarithmic scale
        im = ax.imshow(absH, cmap=colormap, norm=LogNorm(vmin=min_val, vmax=np.max(absH)))
    
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='log(|HProj|)')
    plt.tight_layout()