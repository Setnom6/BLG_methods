import numpy as np
from scipy.linalg import eigh

import matplotlib.pyplot as plt

from enum import Enum

from ManyBodyHamiltonian import ManyBodyHamiltonian

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

class Knothe2024():

    def __init__(self, parameters_to_stablish=None):

        self._initialize_dicts()
        if parameters_to_stablish is not None:
            for key, value in parameters_to_stablish.items():
                if key in self.parameters:
                    self.parameters[key] = value
                else:
                    raise ValueError(f"Parameter {key} is not recognized.")
                
        self.norb = 8
        self.n_elec = 2
        h = self._build_single_particle_dict()
        V = self._build_interaction_dict()
        H = ManyBodyHamiltonian(self.norb, h, V)
        H.generate_basis(self.n_elec)
        self.FSU = H.FSUtils

        self.create_Knothe_basis()
        self.create_singlet_triplet_basis()


    def _initialize_dicts(self):
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
        self.singlet_triplet_correspondence = {
            0: 'LL,S,T0', # Dot, Spin, Valley
            1: 'LL,S,T+', # First 6 are (2,0)
            2: 'LL,S,T-',
            3: 'LL,T0,S',
            4: 'LL,T+,S',
            5: 'LL,T-,S',
            6: 'LR,S,T0', # Next 6 are (1,1) with symmetric spatial part
            7: 'LR,S,T+',
            8: 'LR,S,T-',
            9: 'LR,T0,S',
            10: 'LR,T+,S',
            11: 'LR,T+,S',
            12: 'LR,S,S', # Next 10 are (1,1) with antisymmetruc spatial part
            13: 'LR,T0,T0',
            14: 'LR,T0,T+',
            15: 'LR,T0,T-',
            16: 'LR,T+,T0',
            17: 'LR,T+,T+',
            18: 'LR,T+,T-',
            19: 'LR,T-,T0',
            20: 'LR,T-,T+',
            21: 'LR,T-,T-',
            22: 'RR,S,T0', # Next 6 are (0,2)
            23: 'RR,S,T+', 
            24: 'RR,S,T-',
            25: 'RR,T0,S',
            26: 'RR,T+,S',
            27: 'RR,T+,S',
        }

        self.knothe_correspondence = {
            0: 'S1', 
            1: 'S2', 
            2: 'S3',
            3: 'S4',
            4: 'S5',
            5: 'S6',
            6: 'AS1',
            7: 'AS2',
            8: 'AS3',
            9: 'AS4',
            10: 'AS5',
            11: 'AS6',
            12: 'AS7', 
            13: 'AS8',
            14: 'AS9',
            15: 'AS10',
        }

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
    

    def create_singlet_triplet_basis(self):
        """
        There are 28 basis elements:
        - 16 for (1,1) configurations (all combinations of spin-valley singlet-triplets and viceversa, 8 spatially symmetric and 8 spatially antysimmetric)
        - 8 for (2,0) configurations (combinations of antysimetric spin-valley singlet-triplets) as the spatial wavefunction is symmetric
        - 8 for (0,2) configurations (combinations of antysimetric spin-valley singlet-triplets) as the spatial wavefunction is symmetric
        
        This method takes the basis expressed as integers numbers which represents bit determinants exprresing c_i^dag c_j^dag |0>
        with i,j in 1,..,8 as in orbital_labels and the returns 28 vectors as coeficients expressed in this basis (28 elements per vector) which forms
        the singlet-triplet bases in the spin-valley representation.
        """

        list_of_vectors = []

        # (2,0) configurations
        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([0,3]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,2]): -1},

            {self.FSU.create_state_from_occupied_orbitals([0,1]): +1},

            {self.FSU.create_state_from_occupied_orbitals([2,3]): +1},

            {self.FSU.create_state_from_occupied_orbitals([0,3]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,2]): +1},

            {self.FSU.create_state_from_occupied_orbitals([0,2]): +1},

            {self.FSU.create_state_from_occupied_orbitals([1,3]): +1},
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # (1,1) orbital symmetric configurations

        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([0,7]):+1,
             self.FSU.create_state_from_occupied_orbitals([2,5]):+1,
             self.FSU.create_state_from_occupied_orbitals([1,6]):-1,
             self.FSU.create_state_from_occupied_orbitals([3,4]):-1},

             {self.FSU.create_state_from_occupied_orbitals([0,5]): +1,
              self.FSU.create_state_from_occupied_orbitals([1,4]): -1},

             {self.FSU.create_state_from_occupied_orbitals([2,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([3,6]): -1},

             {self.FSU.create_state_from_occupied_orbitals([0,7]):+1,
             self.FSU.create_state_from_occupied_orbitals([2,5]):-1,
             self.FSU.create_state_from_occupied_orbitals([1,6]):+1,
             self.FSU.create_state_from_occupied_orbitals([3,4]):-1},

             {self.FSU.create_state_from_occupied_orbitals([0,6]): +1,
              self.FSU.create_state_from_occupied_orbitals([2,4]): -1},

             {self.FSU.create_state_from_occupied_orbitals([1,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([3,5]): -1}
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # (1,1) orbital antisymmetric configurations
        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([0,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([2,5]): -1,
            self.FSU.create_state_from_occupied_orbitals([1,6]): -1,
            self.FSU.create_state_from_occupied_orbitals([3,4]): +1},  # 12: S,S

            {self.FSU.create_state_from_occupied_orbitals([0,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([2,5]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,6]): +1,
            self.FSU.create_state_from_occupied_orbitals([3,4]): +1},  # 13: T0,T0

            {self.FSU.create_state_from_occupied_orbitals([0,5]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,4]): +1},  # 14: T0,T+

            {self.FSU.create_state_from_occupied_orbitals([2,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([3,6]): +1},  # 15: T0,T-

            {self.FSU.create_state_from_occupied_orbitals([0,6]): +1,
            self.FSU.create_state_from_occupied_orbitals([2,4]): +1},  # 16: T+,T0

            {self.FSU.create_state_from_occupied_orbitals([0,4]): +1},  # 17: T+,T+

            {self.FSU.create_state_from_occupied_orbitals([2,6]): +1},  # 18: T+,T-

            {self.FSU.create_state_from_occupied_orbitals([1,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([3,5]): +1},  # 19: T-,T0

            {self.FSU.create_state_from_occupied_orbitals([1,5]): +1},  # 20: T-,T+

            {self.FSU.create_state_from_occupied_orbitals([3,7]): +1}   # 21: T-,T-
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # (0,2) configurations

        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([4,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([5,6]): -1},

            {self.FSU.create_state_from_occupied_orbitals([4,5]): +1},

            {self.FSU.create_state_from_occupied_orbitals([6,7]): +1},

            {self.FSU.create_state_from_occupied_orbitals([4,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([5,6]): +1},

            {self.FSU.create_state_from_occupied_orbitals([4,6]): +1},

            {self.FSU.create_state_from_occupied_orbitals([5,7]): +1},
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))


        self.singlet_triplet_basis = list_of_vectors

    def create_Knothe_basis(self):
        """
        There are 16 basis elements:
        - 10 for antysimmetric orbital part
        - 6 for symmetric orbital part
        
        This method takes the basis expressed as integers numbers which represents bit determinants exprresing c_i^dag c_j^dag |0>
        with i,j in 1,..,8 as in orbital_labels and the returns 16 vectors as coeficients expressed in this basis (28 elements per vector) which forms
        the symmetic-antisymmetric basis.
        """

        self.compute_some_characteristic_properties()
        list_of_vectors = []

        # orbital symmetric configurations

        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([0,7]): self.a1/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([3,4]): -self.a1/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([1,6]): self.b*self.a1/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([2,5]): -self.b*self.a1/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([0,3]): self.a2,
             self.FSU.create_state_from_occupied_orbitals([1,2]): self.b*self.a2,
             self.FSU.create_state_from_occupied_orbitals([4,7]): self.a2,
             self.FSU.create_state_from_occupied_orbitals([5,6]): self.b*self.a2}, #S1

             {self.FSU.create_state_from_occupied_orbitals([3,5]): self.a1/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([1,7]): -self.a1/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([5,7]): -self.a2,
             self.FSU.create_state_from_occupied_orbitals([1,3]): -self.a2}, #S2

             {self.FSU.create_state_from_occupied_orbitals([0,6]): self.a1/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([2,4]): -self.a1/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([0,2]): self.a2,
             self.FSU.create_state_from_occupied_orbitals([4,6]): self.a2}, #S3

             {self.FSU.create_state_from_occupied_orbitals([1,4]): self.a1/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([0,5]): -self.a1/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([0,1]): -self.a2,
             self.FSU.create_state_from_occupied_orbitals([4,5]): -self.a2}, #S4

             {self.FSU.create_state_from_occupied_orbitals([2,7]): self.a1/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([3,6]): -self.a1/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([2,3]): self.a2,
             self.FSU.create_state_from_occupied_orbitals([6,7]): self.a2}, #S5

             {self.FSU.create_state_from_occupied_orbitals([1,6]): self.a1/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([2,5]): -self.a1/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([3,4]): self.b*self.a1/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([0,7]): -self.b*self.a1/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([1,2]): self.a2,
             self.FSU.create_state_from_occupied_orbitals([0,3]): -self.b*self.a2,
             self.FSU.create_state_from_occupied_orbitals([5,6]): self.a2,
             self.FSU.create_state_from_occupied_orbitals([4,7]): -self.b*self.a2} #S1
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # orbital antisymmetric configurations
        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([0,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([3,4]): +1}, 

            {self.FSU.create_state_from_occupied_orbitals([3,7]): +1}, 

            {self.FSU.create_state_from_occupied_orbitals([0,4]): +1},  

            {self.FSU.create_state_from_occupied_orbitals([2,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([3,6]): +1},  

            {self.FSU.create_state_from_occupied_orbitals([0,6]): +1,
            self.FSU.create_state_from_occupied_orbitals([2,4]): +1}, 

            {self.FSU.create_state_from_occupied_orbitals([1,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([3,5]): +1}, 

            {self.FSU.create_state_from_occupied_orbitals([0,5]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,4]): +1},   

            {self.FSU.create_state_from_occupied_orbitals([1,6]): +1,
            self.FSU.create_state_from_occupied_orbitals([2,5]): +1},   

            {self.FSU.create_state_from_occupied_orbitals([2,6]): +1},  

            {self.FSU.create_state_from_occupied_orbitals([1,5]): +1},  
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        self.knothe_basis = list_of_vectors

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
        H = ManyBodyHamiltonian(self.norb, h, V)
        H.build(self.n_elec)
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