from many_body_builder import ManyBodyHamiltonian, get_occupied_orbitals
import numpy as np
from scipy.linalg import eigh

import matplotlib.pyplot as plt

from enum import Enum

# TODO: Undertand better how to stablish the singlet-triplet basis
# As the parameter are seteed now, the characteristic quantities indicate which state is predomimant but then the coeffs seems to say a different thing.
# It is necessary to stablish a correspondence between the diagonalized eigenstates and the singlet-triplet basis.
# How to obtain the simmetrized general orbital state?
# DeltaKK and parallel magnetic field are included now, but we dont care about them if we cannot say which state is which.
# The next step is to set realistic values (besides Knothe 2024) and see if the classification is correct.
# The following would be to include the second spatial orbital and add the coupling between them all.


class DQDParameters(Enum):
    B_FIELD = 'b_field'
    B_PARALLEL = 'b_parallel'
    MUB = 'mub'
    GS = 'gs'
    GV = 'gv'
    DELTA_SO = 'DeltaSO'
    DELTA_KK = 'DeltaKK'
    E_I = 'E_i'
    T = 't'
    T_SOC = 't_soc'
    U0 = 'U0'
    U1 = 'U1'
    X = 'X'
    A = 'A'
    P = 'P'
    J = 'J'
    G_ORTHO = 'g_ortho'
    G_ZZ = 'g_zz'
    G_Z0 = 'g_z0'
    G_0Z = 'g_0z'


class DQD_spin_valley_singleOrbital_twoParticles():
    """
    Extended Hubbard Hamiltonian for BLG DQD system according to Knothe 2024.

    Simulates the spin-valley coupling in a double quantum dot system with specific physical effects,
    such as spin-orbit interaction, valley Zeeman effect, and Kane-Mele splitting. It comprises the
    construction of the single-particle Hamiltonian, on-site Coulomb potentials, inter-site interactions,
    and the many-body Hamiltonian. The eigenvalues are calculated for varying input parameters.

    Parameters:
    N : int
        Number of sites in the system.
    """
    def __init__(self, parameters_to_change=None):
        self.parameters = {
            'b_field': 0.0, # Magnetic field in Tesla along the z-axis
            'b_parallel': 0.0, # Magnetic field in Tesla along the x-axis
            'mub': 0.05788, # Bohr magneton in meV/Tesla
            'gs': 2.0, # Spin g-factor
            'gv': 28.0, # Valley g-factor (different for each spatial orbtial)
            'DeltaSO': 0.04, # Kane Mele spin-orbit coupling in meV
            'DeltaKK': 0.0, # Valley mixing in meV
            'E_i': 0.0, # Single-particle energy level in eV expressed as a difference between the two dots
            't': 4.0, # Spin-conserving hopping in meV
            't_soc': 0.0, # Spin-flip hopping in meV due to extrinsic spin-orbit coupling
            'U0': 8.5, # On-site Coulomb potential in meV
            'U1': 1.0, # nearest-neighbour Coulomb potential in meV
            'X': 3.0, # Intersite Exchange interaction in meV
            'A': 0.1, # Density assisted hopping in meV
            'P': 0.02, # Pair hopping interaction in meV
            'J': 0.075, # Renormalization parameter for Coulomb corrections
            'g_ortho': 1.0, #  Orthogonal component of tunneling corrections
            'g_zz': 10.0, # Correction along the z-axis in meV
            'g_z0': -1.0, # Correction for Z-O mixing
            'g_0z': -1.0, # Correction for O-Z mixing
        }

        self.orbital_labels = {
            0: 'LUp+',
            1: 'LDown+',
            2: 'LUp-',
            3: 'LDown-',
            4: 'RUp+',
            5: 'RDown+',
            6: 'RUp-',
            7: 'RDown-'
        }

        self.labels_to_indices = {v: k for k, v in self.orbital_labels.items()}

        if parameters_to_change is not None:
            for key, value in parameters_to_change.items():
                if key in self.parameters:
                    self.parameters[key] = value
                else:
                    raise ValueError(f"Parameter {key} is not recognized.")
                

        self.N = 8 # Number of sites in the system (LUp+, LDown+, LUp-, LDown-, RUp+, RDown+, RUp-, RDown-)
        self.numParticles = 2 # Number of particles in the system (two electrons)
        h = self._build_single_particle_dict()
        V = self._build_interaction_dict()
        H = ManyBodyHamiltonian(self.N, h, V)
        self.basis = H.generate_basis(self.numParticles)  # Generate the basis for two electrons

    def _build_single_particle_dict(self):
        """
        Constructs the single-particle Hamiltonian dictionary for the double quantum dot system.
        The basis is (LUp+, LDown+, LUp-, LDown-, RUp+, RDown+, RUp-, RDown-)
        """
        b_field = self.parameters['b_field']
        b_parallel = self.parameters['b_parallel']
        mub = self.parameters['mub']
        gs = self.parameters['gs']
        gv = self.parameters['gv']
        DeltaSO = self.parameters['DeltaSO']
        DeltaKK = self.parameters['DeltaKK']
        t = self.parameters['t']
        t_soc = self.parameters['t_soc']
        Ei = self.parameters['E_i']

        spin_zeeman_splitting = 0.5 * gs * mub * b_field
        spin_parallel_splitting = 0.5 * gs * mub * b_parallel
        valley_zeeman_splitting = 0.5 * gv * mub * b_field
        kane_mele_splitting = 0.5 * DeltaSO

        # Intradot dynamics
        h = {(0, 0): kane_mele_splitting + valley_zeeman_splitting + spin_zeeman_splitting,  # LUp+
         (1, 1): - kane_mele_splitting + valley_zeeman_splitting - spin_zeeman_splitting,  # LDown+
         (2, 2): - kane_mele_splitting - valley_zeeman_splitting + spin_zeeman_splitting,  # LUp-
         (3, 3): kane_mele_splitting - valley_zeeman_splitting - spin_zeeman_splitting,  # LDown-
         (4, 4): Ei + kane_mele_splitting + valley_zeeman_splitting + spin_zeeman_splitting, # RUp+
         (5, 5): Ei - kane_mele_splitting + valley_zeeman_splitting - spin_zeeman_splitting, # RDown+
         (6, 6): Ei - kane_mele_splitting - valley_zeeman_splitting + spin_zeeman_splitting, # RUp-
         (7, 7): Ei + kane_mele_splitting - valley_zeeman_splitting - spin_zeeman_splitting, # RDown-
         (0, 1): spin_parallel_splitting,
         (2, 3): spin_parallel_splitting,
         (4, 5): spin_parallel_splitting,
         (6, 7): spin_parallel_splitting,
         (0, 2): DeltaKK,
         (1, 3): DeltaKK,
         (4, 6): DeltaKK,
         (5, 7): DeltaKK,
         }

        # Interdot dynamics
        h.update({
            (0, 4): t,  # LUp+ ↔ RUp+
            (1, 5): t,  # LDown+ ↔ RDown+
            (2, 6): t,  # LUp- ↔ RUp-
            (3, 7): t   # LDown- ↔ RDown-
        })

        h.update({
            (0, 5): 1j * t_soc,   # LUp+ → RDown+
            (1, 4): -1j * t_soc,   # LDown+ → RUp+
            (2, 7): 1j * t_soc,    # LUp- → RDown-
            (3, 6): -1j * t_soc,   # LDown- → RUp-
        })


        return h
    
    def _build_interaction_dict(self):
        """
        Constructs the interaction Hamiltonian dictionary for the double quantum dot system.
        Is uses the ordering in the basis given by the many body Hamiltonian.
        """

        U0 = self.parameters['U0']
        U1 = self.parameters['U1']
        X = self.parameters['X']
        A = self.parameters['A']
        P = self.parameters['P']
        J = self.parameters['J']
        g_ortho = self.parameters['g_ortho']
        g_zz = self.parameters['g_zz']
        g_z0 = self.parameters['g_z0']
        g_0z = self.parameters['g_0z']

        
        V = {
        (0, 1, 1, 0): U0 + J*(g_zz + g_z0 + g_0z),
        (2, 3, 3, 2): U0 + J*(g_zz + g_z0 + g_0z),
        (0, 2, 2, 0): U0 + J*(g_zz - g_z0 - g_0z),
        (0, 3, 3, 0): U0 + J*(g_zz - g_z0 - g_0z),
        (1, 2, 2, 1): U0 + J*(g_zz - g_z0 - g_0z),
        (1, 3, 3, 1): U0 + J*(g_zz - g_z0 - g_0z),
        (0, 2, 0, 2): 4*J*g_ortho,
        (0, 3, 1, 2): 4*J*g_ortho,
        (1, 3, 1, 3): 4*J*g_ortho,
        (1, 2, 0, 3): 4*J*g_ortho,
        (4, 5, 5, 4): U0 + J*(g_zz + g_z0 + g_0z),
        (6, 7, 7, 6): U0 + J*(g_zz + g_z0 + g_0z),
        (4, 6, 6, 4): U0 + J*(g_zz - g_z0 - g_0z),
        (4, 7, 7, 4): U0 + J*(g_zz - g_z0 - g_0z),
        (5, 6, 6, 5): U0 + J*(g_zz - g_z0 - g_0z),
        (5, 7, 7, 5): U0 + J*(g_zz - g_z0 - g_0z),
        (4, 6, 4, 6): 4 * J * g_ortho,
        (4, 7, 5, 6): 4 * J * g_ortho,
        (5, 7, 5, 7): 4 * J * g_ortho,
        (5, 6, 4, 7): 4 * J * g_ortho,
        (0, 4, 4, 0): U1,
        (0, 5, 5, 0): U1,
        (0, 6, 6, 0): U1,
        (0, 7, 7, 0): U1,
        (1, 4, 4, 1): U1,
        (1, 5, 5, 1): U1,
        (1, 6, 6, 1): U1,
        (1, 7, 7, 1): U1,
        (2, 4, 4, 2): U1,
        (2, 5, 5, 2): U1,
        (2, 6, 6, 2): U1,
        (2, 7, 7, 2): U1,
        (3, 4, 4, 3): U1,
        (3, 5, 5, 3): U1,
        (3, 6, 6, 3): U1,
        (3, 7, 7, 3): U1,
        (0, 1, 5, 4): P,
        (0, 2, 6, 4): P,
        (0, 3, 7, 4): P,
        (1, 2, 6, 5): P,
        (1, 3, 7, 5): P,
        (2, 3, 7, 6): P,
        (1, 2, 6, 1): A,
        (1, 3, 7, 1): A,
        (2, 3, 7, 2): A,
        (1, 6, 2, 1): A,
        (0, 1, 5, 0): A,
        (0, 2, 6, 0): A,
        (0, 3, 7, 0): A,
        (0, 5, 1, 0): A,
        (0, 6, 2, 0): A,
        (0, 7, 3, 0): A,
        (1, 4, 0, 1): A,
        (1, 7, 3, 1): A,
        (2, 7, 3, 2): A,
        (2, 4, 0, 2): A,
        (2, 5, 1, 2): A,
        (3, 4, 0, 3): A,
        (3, 6, 2, 3): A,
        (3, 5, 1, 3): A,
        }
        exchange_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
        for (r, l) in exchange_pairs:
            if (r, l) < (l, r):
                V[(r, l, r, l)] = X

        return V
    
    
    def classify_determinant(self, state: int) -> dict:
        occupied = get_occupied_orbitals(state, self.N)
        assert len(occupied) == 2
        orb1, orb2 = occupied
        lab1 = self.orbital_labels[orb1]
        lab2 = self.orbital_labels[orb2]

        spin1 = 'Up' if 'Up' in lab1 else 'Down'
        spin2 = 'Up' if 'Up' in lab2 else 'Down'
        valley1 = '+' if '+' in lab1 else '-'
        valley2 = '+' if '+' in lab2 else '-'

        orb_part1 = lab1.replace('Up', '').replace('Down', '').replace('+', '').replace('-', '')
        orb_part2 = lab2.replace('Up', '').replace('Down', '').replace('+', '').replace('-', '')

        # --- SPIN (S) ---
        if spin1 == spin2:
            spin_type = 'Triplet'
        else:
            if orb_part1 == orb_part2 and valley1 == valley2:
                spin_type = 'Singlet'
            else:
                spin_type = 'Mixed'

        # --- VALLEY (V) ---
        if valley1 == valley2:
            valley_type = 'Triplet'
        else:
            if orb_part1 == orb_part2 and spin1 == spin2:
                valley_type = 'Singlet'
            else:
                valley_type = 'Mixed'

        # --- SYMMETRY ---
        if orb_part1 == orb_part2:
            symmetry = 'S'
        elif orb_part1 != orb_part2:
            if (spin_type == 'Singlet' and valley_type == 'Singlet') or (spin_type == 'Triplet' and valley_type == 'Triplet'):
                symmetry = 'AS'
            elif (spin_type == 'Singlet' and valley_type == 'Triplet') or (spin_type == 'Triplet' and valley_type == 'Singlet'): 
                symmetry = 'S'
            else:
                symmetry = 'Mixed'

        return {
            "Orbitals": [lab1, lab2],
            "Valley": valley_type,
            "Spin": spin_type,
            "Orbital Symmetry": symmetry,
        }
    
    def create_determinant_from_labels(self, labels: list) -> int:
        """
        Creates a determinant from a list of orbital labels.
        The labels should be in the format 'LUp+', 'LDown+', etc.
        """
        state = 0
        for label in labels:
            for i, orb_label in self.orbital_labels.items():
                if label == orb_label:
                    state |= (1 << i)
                    break
        return state

    

    def classify_superposition_state(self, coeffs: np.ndarray) -> dict:
        contributions = []
        for i, state in enumerate(self.basis):
            classification = self.classify_determinant(state)
            prob = np.abs(coeffs[i])**2

            contributions.append({
                "state": state,
                "prob": prob,
                "classification": classification
            })
        
        contributions.sort(key=lambda x: -x["prob"])
        
        return {
            "Contributions": contributions,
        }
        

    
    def calculate_eigenvalues_and_eigenvectors(self, parameters_to_change=None):
        """
        Calculates the eigenvalues and eigenvectors of the many-body Hamiltonian for the double quantum dot system.

        Parameters:
        parameters_to_change : dict, optional
            A dictionary of parameters to change in the Hamiltonian. If None, the default parameters are used.

        Returns:
        eigval : ndarray
            The eigenvalues of the many-body Hamiltonian.
        eigv : ndarray
            The eigenvectors of the many-body Hamiltonian.
        """
        if parameters_to_change is not None:
            self.parameters.update(parameters_to_change)


        h = self._build_single_particle_dict()
        V = self._build_interaction_dict()
        H = ManyBodyHamiltonian(self.N, h, V)
        H.build(self.basis)
        eigval, eigv = eigh(H.matrix)

        return eigval, eigv

    
    def compute_some_characteristic_properties(self):
        U0 = self.parameters[DQDParameters.U0.value]
        U1 = self.parameters[DQDParameters.U1.value]
        t = self.parameters[DQDParameters.T.value]
        J = self.parameters[DQDParameters.J.value]
        g_ortho = self.parameters[DQDParameters.G_ORTHO.value]
        g_zz = self.parameters[DQDParameters.G_ZZ.value]
        DeltaSO = self.parameters[DQDParameters.DELTA_SO.value]


        self.a1 = 1.0/np.sqrt(1+16*t**2/(U0-U1+np.sqrt(16*t**2+(U0-U1)**2))**2)
        self.a2 = 1.0/(self.a1*np.sqrt(4+(U0-U1)**2/(4*t**2))*np.sqrt(2))
        self.alpha = 1.0/(3+ (U0-U1)**2/(4*t**2))
        self.DeltaOrb = - (U0-U1)/ 2.0 + 0.5*np.sqrt((U0-U1)**2 + 16*t**2)
        self.DiffOrb = self.DeltaOrb - J *self.alpha* g_zz
        self.DiffIntraOrb = 4*DeltaSO + 8 * J*self.alpha * g_ortho
        self.b = self.alpha * g_ortho * J / DeltaSO
        self.C = np.sqrt(1+self.b**2)

    

def represent_B_field_dependence(arrayToPlot, fixedParameters=None, number_of_eigenstates=16):
    dqd = DQD_spin_valley_singleOrbital_twoParticles()

    bFields = arrayToPlot if arrayToPlot is not None else np.linspace(-0.5, 0.5, 100)  # Default range for B-field
    eigvals = np.zeros((len(bFields), len(dqd.basis)), dtype=np.complex128)
    eigvectors = np.zeros((len(bFields), len(dqd.basis), len(dqd.basis)), dtype=np.complex128)

    for i, bField in enumerate(bFields):
        if bField < 0:
            parameters_to_change = fixedParameters.copy() if fixedParameters is not None else {}
            parameters_to_change[DQDParameters.B_PARALLEL.value] = abs(bField)
            eigval, eigv = dqd.calculate_eigenvalues_and_eigenvectors(parameters_to_change= parameters_to_change)
        else:
            parameters_to_change = fixedParameters.copy() if fixedParameters is not None else {}
            parameters_to_change[DQDParameters.B_FIELD.value] = bField
            eigval, eigv = dqd.calculate_eigenvalues_and_eigenvectors( parameters_to_change= parameters_to_change)

        eigvals[i] = eigval
        eigvectors[i] = eigv

    #reordered_eigvals, _ = reorder_eigenvectors_by_overlap(eigvals, eigvectors)

    plt.figure()
    for i in range(number_of_eigenstates):
        plt.plot(bFields, eigvals[:, i], label=f'Eigenstate {i+1}')
    plt.xlabel('B-field (T)')
    plt.ylabel('Eigenvalue (meV)')
    plt.title('Eigenvalues of DQD Spin-Valley System')
    plt.grid(True)
    plt.show()


import numpy as np

def reorder_eigenvectors_by_overlap(list_of_eigenvalues, list_of_eigenvectors):
    """
    Reorders eigenvalues and eigenvectors based on the overlap between consecutive eigenvectors.
    This ensures continuity in eigenvalue trajectories (e.g., for tracking crossings/anti-crossings).

    Parameters:
    -----------
    list_of_eigenvalues : list of np.ndarray
        List of 1D arrays containing eigenvalues for each step (shape: [n_steps, n_eigenvalues]).
    list_of_eigenvectors : list of np.ndarray
        List of 2D arrays where each column is an eigenvector (shape: [n_steps, n_dim, n_eigenvalues]).

    Returns:
    --------
    reordered_eigenvalues : np.ndarray
        Reordered eigenvalues (shape: [n_steps, n_eigenvalues]).
    reordered_eigenvectors : np.ndarray
        Reordered eigenvectors (shape: [n_steps, n_dim, n_eigenvalues]).
    """
    # Convert inputs to numpy arrays
    eigenvalues = np.array(list_of_eigenvalues)
    eigenvectors = np.array(list_of_eigenvectors)
    n_steps, n_dim, n_eigenvalues = eigenvectors.shape

    # Initialize output arrays
    reordered_eigenvalues = np.zeros_like(eigenvalues)
    reordered_eigenvectors = np.zeros_like(eigenvectors)
    reordered_eigenvalues[0] = eigenvalues[0]  # First step is unchanged
    reordered_eigenvectors[0] = eigenvectors[0]

    for i in range(1, n_steps):
        # Compute overlap between current and previous (reordered) eigenvectors
        overlap = np.abs(reordered_eigenvectors[i-1].T @ eigenvectors[i])

        # Track assigned indices to avoid duplicates
        assigned_indices = set()
        for j in range(n_eigenvalues):
            # Find unassigned eigenvector with maximum overlap
            max_overlap_idx = -1
            max_overlap = -1
            for k in range(n_eigenvalues):
                if k not in assigned_indices and overlap[j, k] > max_overlap:
                    max_overlap = overlap[j, k]
                    max_overlap_idx = k

            # Assign the best match
            if max_overlap_idx != -1:
                reordered_eigenvalues[i, j] = eigenvalues[i, max_overlap_idx]
                reordered_eigenvectors[i, :, j] = eigenvectors[i, :, max_overlap_idx]
                assigned_indices.add(max_overlap_idx)

    return reordered_eigenvalues, reordered_eigenvectors


def represent_detuning_dependence(arrayToPlot, fixedParameters=None, number_of_eigenstates=16):
    dqd = DQD_spin_valley_singleOrbital_twoParticles()

    eps = arrayToPlot if arrayToPlot is not None else np.linspace(-0.5, 0.5, 100)  # Default range for detuning
    eigvals = np.zeros((len(eps), len(dqd.basis)), dtype=np.complex128)

    for i, eps_i in enumerate(eps):
        parameters_to_change = fixedParameters.copy() if fixedParameters is not None else {}
        parameters_to_change[DQDParameters.E_I.value] = eps_i
        eigval, eigv = dqd.calculate_eigenvalues_and_eigenvectors(parameters_to_change=parameters_to_change)
        eigvals[i] = eigval


    plt.figure()
    for i in range(number_of_eigenstates):
        plt.plot(eps, eigvals[:, i], label=f'Eigenstate {i+1}')
    plt.xlabel('E_i (meV)')
    plt.ylabel('Eigenvalue (meV)')
    plt.title('Eigenvalues of DQD Spin-Valley System')
    plt.show()

def represent_DeltaSO_dependence(arrayToPlot, fixedParameters=None, number_of_eigenstates=16):
    dqd = DQD_spin_valley_singleOrbital_twoParticles()

    DeltaSO_values = arrayToPlot if arrayToPlot is not None else np.linspace(0.0, 0.1, 100)  # Default range for DeltaSO
    eigvals = np.zeros((len(DeltaSO_values), len(dqd.basis)), dtype=np.complex128)
    eigvectors = np.zeros((len(DeltaSO_values), len(dqd.basis), len(dqd.basis)), dtype=np.complex128)

    for i, DeltaSO in enumerate(DeltaSO_values):
        parameters_to_change = fixedParameters.copy() if fixedParameters is not None else {}
        parameters_to_change[DQDParameters.DELTA_SO.value] = DeltaSO
        eigval, eigv = dqd.calculate_eigenvalues_and_eigenvectors(parameters_to_change=parameters_to_change)
        eigvals[i] = eigval
        eigvectors[i] = eigv

    #reordered_eigvals, _ = reorder_eigenvectors_by_overlap(eigvals, eigvectors)
    plt.figure()
    for i in range(number_of_eigenstates):
        plt.plot(DeltaSO_values, eigvals[:, i], label=f'Eigenstate {i+1}')
    plt.xlabel('Delta SO (meV)')
    plt.ylabel('Eigenvalue (meV)')
    plt.title('Eigenvalues of DQD Spin-Valley System')
    plt.grid(True)
    plt.show()

def zero_field_classification(fixedParameters=None, number_of_eigenstates=16):
    dqd = DQD_spin_valley_singleOrbital_twoParticles()
    parameters_to_change = fixedParameters.copy() if fixedParameters is not None else {}
    parameters_to_change[DQDParameters.B_FIELD.value] = 0.1  # Set a small B-field to avoid degeneracy
    eigval, eigv = dqd.calculate_eigenvalues_and_eigenvectors(fixedParameters)

    print([bin(state) for state in dqd.basis])

    dqd.compute_some_characteristic_properties()
    print(f"Delta Orbital = {dqd.DeltaOrb:.5f} meV")
    print(f"Diff Orbital = {dqd.DiffOrb:.5f} meV")
    print(f"Diff Intra Orbital = {dqd.DiffIntraOrb:.5f} meV")
    print(f"a1 = {dqd.a1:.5f}")
    print(f"a2 = {dqd.a2:.5f}")
    print(f"alpha = {dqd.alpha:.5f}")
    print(f"b = {dqd.b:.5f}")
    print(f"C = {dqd.C:.5f}")
    print("\n")
    
    for i in range(number_of_eigenstates):
        classification = dqd.classify_superposition_state(eigv[:, i])
        print(f"Eigenstate {i+1}:")
        print(f"Eigenvalue: {eigval[i]:.4f} meV")
        print("Contributions:")
        for contrib in classification['Contributions']:
            if contrib['prob'] > 1e-5:
                print(f"  State: {bin(contrib['state'])}, Probability: {contrib['prob']:.4f}, Classification: {contrib['classification']}")
        print("\n")

if __name__ == "__main__":

    gOrtho = 10
    U0 = 8.5
    # Example usage
    fixedParameters = {
        DQDParameters.B_FIELD.value: 0.00,  # Set B-field to zero for initial classification
        DQDParameters.B_PARALLEL.value: 0.0,  # Set parallel magnetic field to zero
        DQDParameters.E_I.value: 00.0,  # Set detuning to
        DQDParameters.T.value: 0.04,  # Set hopping parameter
        DQDParameters.DELTA_SO.value: -0.04,  # Set Kane
        DQDParameters.DELTA_KK.value: 0.02,  # Set valley mixing
        DQDParameters.T_SOC.value: 0.0,  # Set spin-flip
        DQDParameters.U0.value: U0,  # Set on-site Coul
        DQDParameters.U1.value: 5.0,  # Set nearest-neighbour Coulomb potential
        DQDParameters.X.value: 0.02,  # Set intersite exchange interaction
        DQDParameters.G_ORTHO.value: gOrtho,  # Set orthogonal component of tunneling corrections
        DQDParameters.G_ZZ.value:10*gOrtho,  # Set correction along
        DQDParameters.G_Z0.value: 2*gOrtho/3,  # Set correction for Z-O mixing
        DQDParameters.G_0Z.value: 2*gOrtho/3,  # Set correction for O-Z mixing
        DQDParameters.GS.value: 2.0,  # Set spin g-factor
        DQDParameters.GV.value: 28.0,  # Set valley g-factor
        DQDParameters.A.value: 0.1,  # Set density assisted hopping
        DQDParameters.P.value: 0.02,  # Set pair hopping interaction
        DQDParameters.J.value: 0.075/gOrtho,  # Set renormalization parameter for Coulomb corrections
    }

    number_of_eigenstates = 22  # Number of eigenstates to plot
    arrayToPlot = np.linspace(3.0, 5.0, 1000)  # Example range for B-field or detuning
    zero_field_classification(fixedParameters=fixedParameters, number_of_eigenstates=number_of_eigenstates)
    represent_detuning_dependence(arrayToPlot=arrayToPlot, fixedParameters=fixedParameters, number_of_eigenstates=number_of_eigenstates)
    
    
        