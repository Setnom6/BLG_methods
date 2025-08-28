from src.DQD_2particles_1orbital import DQD_2particles_1orbital, DQDParameters
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from matplotlib.colors import LogNorm
from scipy.linalg import eigh, orth
from matplotlib.cm import get_cmap
from typing import Dict
from matplotlib.patches import Patch
from enum import Enum
from joblib import Parallel, delayed


class BasisToProject(Enum):
    SINGLET_TRIPLET_BASIS = 'Singlet Triplet symmetry'
    SINGLET_TRIPLET_IN_SPIN_BASIS = "Singlet-Triplet in Spin"
    SINGLET_TRIPLET_REORDERED_BASIS = 'Singlet Triplet Reordered'
    SPIN_PLUS_VALLYE_MINUS_BASIS = "Spin Plus - Valley Minus"
    ORBITAL_SYMMETRY = 'Orbital symmetry'
    SPIN_SYMMETRY = 'Spin Symmetry'
    VALLEY_SYMMETRY = 'Valley Symmetry'
    TOTAL_SPIN = 'Total Spin'
    TOTAL_VALLEY = 'Total Valley'
    TOTAL_DOT = 'Total Dot'
    ORIGINAL = 'Fock'
    MINIMAL_BASIS = 'Minimal'


class TypeOfPlot(Enum):
    SYMMETRY = "Symmetry"
    PROJECTION = "Projection"
    HEATMAP = "Heatmap"

class PlotParameters(Enum):
    TYPE = "type"
    NUMBER_OF_EIGENSTATES = "number_of_eigenstates"
    FIXED_PARAMETERS = "fixed_parameters"
    SHOW = "show"
    EXTRA_FOLDER = "extra_folder"
    PARAM_TO_ITER = "param_to_iter"
    ARRAY = "array"
    SYMMETRY_COLOR = "symmetry_color"
    ROHLING_STATES = "rohling_states"
    Y_LIMS = "y_lims"
    BASIS = "basis"
    BLOCKS = "blocks"
    SPECIAL_TITLE = "special_title"


class PlotsManager:

    def __init__(self, parametersDict: Dict):
        self.type_of_plot = parametersDict.get(PlotParameters.TYPE, TypeOfPlot.SYMMETRY)
        self.number_of_eigenstates = parametersDict.get(PlotParameters.NUMBER_OF_EIGENSTATES, 28)
        self.fixed_parameters = parametersDict.get(PlotParameters.FIXED_PARAMETERS, {})
        self.show = parametersDict.get(PlotParameters.SHOW, True)
        self.extra_folder = parametersDict.get(PlotParameters.EXTRA_FOLDER, "")
        self.special_title = parametersDict.get(PlotParameters.SPECIAL_TITLE, None)
        self._initialize_dicts()

        if self.type_of_plot == TypeOfPlot.SYMMETRY:
            self.parameter_to_iter = parametersDict.get(PlotParameters.PARAM_TO_ITER, DQDParameters.E_I)
            self.array_to_plot  = parametersDict.get(PlotParameters.ARRAY, np.linspace(0.0, 10.0, 100))
            self.symmetry_color = parametersDict.get(PlotParameters.SYMMETRY_COLOR, BasisToProject.ORIGINAL)
            self.rohling_initial_states = parametersDict.get(PlotParameters.ROHLING_STATES, False)
            self.ylims = parametersDict.get(PlotParameters.Y_LIMS, None)
            self.basis_to_project = None
            self.blocks = None

        elif self.type_of_plot == TypeOfPlot.PROJECTION:
            self.parameter_to_iter = parametersDict.get(PlotParameters.PARAM_TO_ITER, DQDParameters.E_I)
            self.array_to_plot  = parametersDict.get(PlotParameters.ARRAY, np.linspace(0.0, 10.0, 100))
            self.basis_to_project = parametersDict.get(PlotParameters.BASIS, BasisToProject.SINGLET_TRIPLET_BASIS)
            self.ylims = parametersDict.get(PlotParameters.Y_LIMS, None)
            self.symmetry_color = None
            self.rohling_initial_states = None
            self.blocks = None

        elif self.type_of_plot == TypeOfPlot.HEATMAP:
            self.basis_to_project = parametersDict.get(PlotParameters.BASIS, BasisToProject.SINGLET_TRIPLET_BASIS)
            self.blocks = parametersDict.get(PlotParameters.BLOCKS, None)
            self.parameter_to_iter = None
            self.array_to_plot = None
            self.symmetry_color = None
            self.rohling_initial_states = None
            self.ylims = None

        else:
            raise AttributeError("TypeOfPlot not supported")

    def _initialize_dicts(self):
        dqd = DQD_2particles_1orbital()
        self.basis_and_correspondences = {
            BasisToProject.ORIGINAL : (dqd.original_basis, dqd.original_correspondence, {}),
            BasisToProject.SINGLET_TRIPLET_BASIS : (dqd.singlet_triplet_basis, dqd.singlet_triplet_correspondence, {"LL": 6, "LR": 16, "RR": 6}),
            BasisToProject.SINGLET_TRIPLET_IN_SPIN_BASIS : (dqd.singlet_triplet_in_spin_basis, dqd.singlet_triplet_in_spin_correspondence, {"LL": 6, "LR--": 4, "LR+-": 8, "LR++": 4, "RR": 6}),
            BasisToProject.SPIN_PLUS_VALLYE_MINUS_BASIS : (dqd.spinPlus_valleyMinus_basis, dqd.spinPlus_valleyMinus_correspondence, {"LL": 6, "LRT-": 4, "LRS": 4, "LRT0": 4,"LRT+": 4, "RR": 6}),
            BasisToProject.SINGLET_TRIPLET_REORDERED_BASIS : (dqd.singlet_triplet_reordered_basis, dqd.singlet_triplet_reordered_correspondence, {"LessEnergetic": 5, "Interacts": 6, "Rest": 17}),
            BasisToProject.MINIMAL_BASIS : (dqd.singlet_triplet_minimal_basis, dqd.singlet_tirplet_minimal_correspondence, {"LessEnergetic": 4, "Interacts": 6, "Rest": 20}),
            BasisToProject.ORBITAL_SYMMETRY:  (dqd.orbital_symmetry_basis, dqd.symmetric_antisymmetric_correspondence, {"Sym": 6, "AntiSym": 10, "ortho": 12}),
            BasisToProject.SPIN_SYMMETRY: (dqd.spin_symmetry_basis, dqd.symmetric_antisymmetric_correspondence, {"Sym": 6, "AntiSym": 10, "ortho": 12}),
            BasisToProject.VALLEY_SYMMETRY: (dqd.valley_symmetry_basis, dqd.symmetric_antisymmetric_correspondence, {"Sym": 6, "AntiSym": 10, "ortho": 12}),
            BasisToProject.TOTAL_DOT: (None, "dot", None),
            BasisToProject.TOTAL_VALLEY: (None, "valley", None),
            BasisToProject.TOTAL_SPIN: (None, "spin", None)
        }

    def plotSimulation(self):
        if self.type_of_plot == TypeOfPlot.SYMMETRY:
            print("Plotting symmetry colored simulation...")
            self._plot_symmetry()
        
        elif self.type_of_plot == TypeOfPlot.PROJECTION:
            print("Plotting projected simulation...")
            self._plot_projection()

        elif self.type_of_plot == TypeOfPlot.HEATMAP:
            print("Plotting heatmap hamiltonian...")
            self._plot_heatmap()

        else:
            pass


    def obtain_dict_parameters_to_change(self, value: float):
        new_dict = self.fixed_parameters.copy()

        if self.parameter_to_iter == DQDParameters.B_FIELD:
            if value < 0:
                new_dict[DQDParameters.B_FIELD.value] = 0.0
                new_dict[DQDParameters.B_PARALLEL.value] = -value
            else:
                new_dict[DQDParameters.B_FIELD.value] = value
                new_dict[DQDParameters.B_PARALLEL.value] = 0.0

        else:
            new_dict[self.parameter_to_iter.value] = value
        
        return new_dict
    
    def obtain_iteration_dict_labels(self):
        dict_labels = {}
        dict_labels["y axis"] = "Eigvalue (meV)" 
        
        if self.parameter_to_iter == DQDParameters.E_I:
            dict_labels["x_axis"] = "E_R (meV)"
            dict_labels["y axis"] = "Eigvalue - E_ref (meV)"

        elif self.parameter_to_iter == DQDParameters.B_FIELD:
            dict_labels["x_axis"] = "B (T)"

        else:
            dict_labels["x_axis"] = self.parameter_to_iter.value

        dict_labels["title"] = "Energy levels"

        if self.symmetry_color is not None:
            dict_labels["colorbar"] = f'{self.symmetry_color.value}: blue (S) to red (AS)'
            dict_labels["title"] = "Energy levels colored by symmetry classification"

            if self.basis_and_correspondences[self.symmetry_color][0] is None: # If it does not have a basis it is because is a total (dot/SPIN/VALLEY)
                if self.symmetry_color.value == BasisToProject.TOTAL_DOT.value:
                    dict_labels["colorbar"] = 'blue (orbitally symmetric) to red (orbitally antisymmetric)'
                else:
                    dict_labels["colorbar"] = f'{self.symmetry_color.value}: blue (triplet) to red (singlet)'

        elif self.basis_to_project is not None:
            dict_labels["title"] = f"Energy levels colored by similarity with {self.basis_to_project.value} states"

        if self.special_title is not None:
            dict_labels["title"] = self.special_title

        return dict_labels
    
    def save_figure_and_parameters(self):
        figures_dir = os.path.join(os.getcwd(),"DQD_2particles_1orbital_methods", "figures", self.extra_folder)
        os.makedirs(figures_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        fig_path = os.path.join(figures_dir, f"{self.type_of_plot.value}_{timestamp}.png")
        plt.savefig(fig_path)
        print(f"Figure saved in: {fig_path}")

        param_path = os.path.join(figures_dir, f"parameters_{self.type_of_plot.value}_{timestamp}.txt")
        with open(param_path, 'w') as f:
            for key, value in self.fixed_parameters.items():
                f.write(f"{key}: {value}\n")
        print(f"Parameters saved in: {param_path}")


    def _plot_symmetry(self):
        dqd = DQD_2particles_1orbital(self.fixed_parameters)

        def compute_eigen_and_color(value):
            parameters_to_change = self.obtain_dict_parameters_to_change(value)
            eigval, eigv = dqd.calculate_eigenvalues_and_eigenvectors(parameters_to_change=parameters_to_change)
            eigvals_local = eigval[:self.number_of_eigenstates].real
            colors_local = np.array([
                self.obtain_colors_for_symmetry_plot(dqd, eigv[:, j])
                for j in range(self.number_of_eigenstates)
            ])
            return eigvals_local, colors_local

        results = Parallel(n_jobs=-1)(
            delayed(compute_eigen_and_color)(value)
            for value in self.array_to_plot
        )

        eigvals = np.array([res[0] for res in results])
        colors = np.array([res[1] for res in results])

        plt.figure(figsize=(10, 6))
        if self.parameter_to_iter == DQDParameters.E_I:
            for j in range(self.number_of_eigenstates):
                plt.scatter(self.array_to_plot, eigvals[:, j] - self.array_to_plot, c=colors[:, j], cmap='bwr', vmin=-1, vmax=1, s=1)
        else:
            for j in range(self.number_of_eigenstates):
                plt.scatter(self.array_to_plot, eigvals[:, j], c=colors[:, j], cmap='bwr', vmin=-1, vmax=1, s=1)

        if self.rohling_initial_states:
            for i in range(16):
                mask = self.array_to_plot <= 1.0
                plt.plot(self.array_to_plot[mask], self.predicted11state(dqd, self.array_to_plot[mask], i))

        labels_dict = self.obtain_iteration_dict_labels()
        plt.xlabel(labels_dict["x_axis"])
        plt.ylabel(labels_dict["y axis"])
        plt.title(labels_dict["title"])
        plt.colorbar(label=labels_dict["colorbar"])
        plt.grid(True)
        plt.tight_layout()
        if self.ylims is not None:
            plt.ylim(self.ylims[0], self.ylims[1])
        self.save_figure_and_parameters()
        if self.show:
            plt.show()
        else:
            plt.close()



    
    def _plot_projection(self):
        dqd = DQD_2particles_1orbital(self.fixed_parameters)
        vectors_list, correspondence, _ = self.basis_and_correspondences[self.basis_to_project]

        def compute_projected_eigen_and_color(value):
            parameters_to_change = self.obtain_dict_parameters_to_change(value)
            projectedH = dqd.project_hamiltonian(vectors_list, parameters_to_change=parameters_to_change)
            eigval, eigv = eigh(projectedH)
            eigvals_local = eigval[:self.number_of_eigenstates].real
            color_data_local = [
                self.obtain_colors_to_plot_in_projection(dqd, correspondence, eigv[:, j])
                for j in range(self.number_of_eigenstates)
            ]
            return eigvals_local, color_data_local

        results = Parallel(n_jobs=-1)(
            delayed(compute_projected_eigen_and_color)(value)
            for value in self.array_to_plot
        )

        eigvals = np.array([res[0] for res in results])
        color_data = [res[1] for res in results]

        base_cmap = get_cmap('tab20')
        color_palette = [base_cmap(i / 20) for i in range(20)]

        fig, ax = plt.subplots(figsize=(10, 6))
        unique_colors_used = {}

        for j in range(self.number_of_eigenstates):
            x = []
            y = []
            c = []
            for i in range(len(self.array_to_plot)):
                x.append(self.array_to_plot[i])
                val = eigvals[i, j] - self.array_to_plot[i] if self.parameter_to_iter == DQDParameters.E_I else eigvals[i, j]
                y.append(val)

                color_idx = color_data[i][j]['color']
                intensity = color_data[i][j]['intensity']
                intensity = max(0.0, min(1.0, intensity))
                base_color = color_palette[color_idx % 20]
                rgba = (
                    base_color[0],
                    base_color[1],
                    base_color[2],
                    1.0  # Visualización más clara: opacidad fija
                )
                c.append(rgba)

                if color_idx not in unique_colors_used and correspondence is not None:
                    unique_colors_used[color_idx] = rgba

            ax.scatter(x, y, color=c, s=1)

        labels_dict = self.obtain_iteration_dict_labels()
        ax.set_xlabel(labels_dict["x_axis"])
        ax.set_ylabel(labels_dict["y axis"])
        ax.set_title(labels_dict["title"])
        ax.grid(True)

        if correspondence is not None and len(unique_colors_used) > 0:
            handles = []
            for color_idx, rgba in sorted(unique_colors_used.items()):
                label = correspondence.get(color_idx, f"ID {color_idx}")
                handles.append(Patch(facecolor=rgba, edgecolor='black', label=label))

            ax.legend(
                handles=handles,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                title='State similarity',
                frameon=False
            )

        plt.tight_layout()
        if self.ylims is not None:
            plt.ylim(self.ylims[0], self.ylims[1])
        self.save_figure_and_parameters()
        if self.show:
            plt.show()
        else:
            plt.close()


    def _plot_heatmap(self):
        dqd = DQD_2particles_1orbital(self.fixed_parameters)
        basis_to_project, _, blocks_dict = self.basis_and_correspondences[self.basis_to_project]

        H_full = dqd.obtain_hamiltonian_determinant_basis(parameters_to_change=self.fixed_parameters)
        dimFull = H_full.shape[0]
        I  = np.eye(dimFull)
        U_P = np.array(basis_to_project).T

        if U_P.shape[1] < 28:
                P_proj = U_P @ U_P.conj().T  # (28,28)
                Q_proj = I - P_proj
                U_Q = orth(Q_proj)

                extra_basis = [vector for vector in U_Q.T]

                total_basis_to_project = basis_to_project + extra_basis

        else:
                total_basis_to_project = basis_to_project

        HProjectedTotal = dqd.project_hamiltonian(total_basis_to_project, parameters_to_change=self.fixed_parameters)
        dqd.diagnoseProjectionQuality(total_basis_to_project, self.fixed_parameters)
        print("-----------------------------\n")

        absH = np.abs(HProjectedTotal)
        fig, ax = plt.subplots(figsize=(8, 8))

        # Set a floor value of 1e-5 for the logarithmic scale
        min_val = 1e-5
        absH[absH < min_val] = min_val  # Replace values smaller than 1e-5 with 1e-5
            
        # Create the image with logarithmic scale
        im = ax.imshow(absH, cmap='viridis', norm=LogNorm(vmin=min_val, vmax=np.max(absH)))

        # Add separations for charge sectors

        sectorSizes = blocks_dict
        if sectorSizes is None:
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
        title = f"|HProj| in {self.basis_to_project.value} basis"
        if self.special_title is not None:
            title = self.special_title

        ax.set_title(title)
        fig.colorbar(im, ax=ax, label='log(|HProj|)')
        plt.tight_layout()
        self.save_figure_and_parameters()
        if self.show:
            plt.show()
        else:
            plt.close()


    def obtain_colors_for_symmetry_plot(self, dqd: DQD_2particles_1orbital, eigenstate: np.ndarray):
        basis, correspondence, _ = self.basis_and_correspondences[self.symmetry_color]

        if basis is None:
            return self.total_dof_color(dqd, correspondence, eigenstate)
        
        if self.symmetry_color == BasisToProject.SINGLET_TRIPLET_BASIS:
            classification = dqd.FSU.classify_eigenstate(basis, correspondence, eigenstate)
            return self.singlet_triplet_symmetry_color(classification, correspondence)
        
        if (self.symmetry_color == BasisToProject.ORBITAL_SYMMETRY
             or self.symmetry_color == BasisToProject.VALLEY_SYMMETRY 
             or self.symmetry_color == BasisToProject.SPIN_SYMMETRY):
            classification = dqd.FSU.classify_eigenstate(basis, correspondence, eigenstate)
            return self.symmetric_antisymmetric_color_difference(classification)
        
        return None
    
    def obtain_colors_to_plot_in_projection(self, dqd: DQD_2particles_1orbital, correspondence: dict, eigenstate: np.ndarray):
        lenBasis = len(eigenstate)
        inv_correspondence = {v: k for k, v in correspondence.items()}
        preferred_basis = []
        for i in range(lenBasis):
            preferred_basis.append(np.array([0]*i+[1]*1+[0]*(lenBasis-i-1)))
        classification = dqd.FSU.classify_eigenstate(preferred_basis, correspondence, eigenstate)
        return {'color': inv_correspondence[classification['most_similar_state']], 'intensity': classification['probability']}

    def symmetric_antisymmetric_color_difference(self, classification: dict):
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

    def singlet_triplet_symmetry_color(self, classification: dict, correspondence: dict):
        inv_correspondence = {v: k for k, v in correspondence.items()}
        antisymm_mask = [12,13,14,15,16,17,18,19,20,21]
        most_probably_state = classification['most_similar_state']
        probability_most_likely = classification['probability']

        AS_weights = 0.0
        S_weights = 0.0
        if inv_correspondence[most_probably_state] in antisymm_mask:
            AS_weights += probability_most_likely
        else:
            S_weights += probability_most_likely

        i = 1
        while (AS_weights+S_weights) < 0.85 and i < 16:
            new_prob = classification['ordered_probabilities'][i]['probability']

            if inv_correspondence[most_probably_state] in antisymm_mask:
                AS_weights+=new_prob
            else:
                S_weights+=new_prob
            i+=1


        normalized_AS_weights = AS_weights / (AS_weights+S_weights)
        normalized_S_weights = S_weights / (AS_weights+S_weights)

        return -1*normalized_S_weights + 1*normalized_AS_weights


    def total_dof_color(self, dqd: DQD_2particles_1orbital, dof: str, state: np.ndarray):
        """
        We identify <S2>=1 (singlet) with value +1 red and <S2>=2 (triplet) with value -1 blue
        """
        S2 = dqd.buildS2Matrix(dof)
        total_spin = dqd.expectationValue(S2, state)
        total_spin = total_spin*2-3
        return -total_spin
    
    def predicted11state(self, dqd: DQD_2particles_1orbital, array_to_plot, index):
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

        return np.array([value for _ in range(len(array_to_plot))])