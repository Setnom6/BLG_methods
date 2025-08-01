
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

    elif basis_to_project == BasisToProject.SINGLET_TRIPLET_REORDERED_BASIS:
        projectedH = dqd.project_hamiltonian(dqd.singlet_triplet_reordered_basis, parameters_to_change=fixed_parameters)
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


def plot_hamiltonian_charge_configuration_blocks(absH: np.ndarray, title: str = "Hamiltonian divided by charge configuration", colormap: str = "viridis", alternative_blocks = None):
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

    if alternative_blocks is None:
        sectorSizes = {
        '(2,0)': 6,
        '(1,1)': 16,
        '(0,2)': 6
        }

    else:
        sectorSizes = alternative_blocks

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

def plot_hamiltonian_separated_blocks(absH, blocks_dict, title = "Hamiltonian", colormap= 'viridis'):
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
    sectorSizes = blocks_dict

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