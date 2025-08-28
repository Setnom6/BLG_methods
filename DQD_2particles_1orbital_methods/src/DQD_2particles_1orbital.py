import numpy as np
from scipy.linalg import eigh

from enum import Enum

from .ManyBodyHamiltonian import ManyBodyHamiltonian

class DQDParameters(Enum):
    B_FIELD = 'b_field'
    B_PARALLEL = 'b_parallel'
    MUB = 'mub'
    GS = 'gs'
    GSLFACTOR = 'gsLeftFactor'
    GV = 'gv'
    GVLFACTOR = 'gvLeftFactor'
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

class DQD_2particles_1orbital():

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

        self._initialize_original_dict()
        self.call_basis_creators()


    def _initialize_dicts(self):
        self.parameters = {
            'b_field': 0.0, # Magnetic field in Tesla along the z-axis
            'b_parallel': 0.0, # Magnetic field in Tesla along the x-axis
            'mub': 0.05788, # Bohr magneton in meV/Tesla
            'gs': 2.0, # Spin g-factor
            'gsLeftFactor': 1.0, # gsL = gsLeftFactor * gs (right)
            'gv': 28.0, # Valley g-factor (different for each spatial orbital)
            'gvLeftFactor': 1.0, # gvL = gvLeftFactor * gv (right)
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
            11: 'LR,T-,S',
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
            27: 'RR,T-,S',
        }

        self.singlet_triplet_in_spin_correspondence = {
            0: 'LL,S,--', # Dot, Spin, Valley, we consider only singlet-triplet in spin an ordered assuming energy order: Up > Down / + > - / valley > spin
            1: 'LL,T-,+-', # First 6 are (2,0)
            2: 'LL,S,+-',
            3: 'LL,T0,+-',
            4: 'LL,T+,+-',
            5: 'LL,S,++',
            6: 'LR,T-,--', # Next 4 are (1,1) with valleys --
            7: 'LR,S,--',
            8: 'LR,T0,--',
            9: 'LR,T+,--',
            10: 'LR,T-,-+', # Next 8 are (1,1) with valleys +. or -+
            11: 'LR,T-,+-',
            12: 'LR,S,-+', 
            13: 'LR,S,+-',
            14: 'LR,T0,-+',
            15: 'LR,T0,+-',
            16: 'LR,T+,-+',
            17: 'LR,T+,+-',
            18: 'LR,T-,++', # Next 4 are (1,1) with valleys ++
            19: 'LR,S,++',
            20: 'LR,T0,++',
            21: 'LR,T+,++',
            22: 'RR,S,--', # Next 6 are (0,2)
            23: 'RR,T-,+-', 
            24: 'RR,S,+-',
            25: 'RR,T0,+-',
            26: 'RR,T+,+-',
            27: 'RR,S,++',
        }

        self.singlet_triplet_reordered_correspondence = {
            0: 'LR,T+,T-', # We aislee the states which interact with the first 5 states (when Ei grows positively)
            1: 'LR,S,T-', 
            2: 'LL,S,T-',
            3: 'LR,T0,T-',
            4: 'LR,T-,T-',
            5: 'LR,T+,T0', #  Interacts with 0
            6: 'LL,S,T0', # Interacts with 2
            7: 'LR,S,T0', # Interacts with 1
            8: 'RR,S,T-', # Interacts with 1 and 2
            9: 'LR,T0,T0', #Interacts with 4
            10: 'LR,T-,T0', # Interacts with 4
            11: 'LL,T-,S', # Higher energy states
            12: 'LL,S,T+',
            13: 'LR,S,T+',
            14: 'LR,T0,T+',
            15: 'LR,T+,T+',
            16: 'LL,T0,S', 
            17: 'LL,T+,S',
            18: 'LR,T0,S',
            19: 'LR,T-,S',
            20: 'LR,T+,S',
            21: 'LR,S,S',
            22: 'LR,T-,T+',
            23: 'RR,T-,S', 
            24: 'RR,T0,S',
            25: 'RR,T+,S',
            26: 'RR,S,T0',
            27: 'RR,S,T+',
        }

        self.singlet_tirplet_minimal_correspondence = {
            0: 'LR,T-,T-', # The working basis are the first 3 states
            1: 'LR,S,T-', 
            2: 'LL,S,T-',
            3: 'LR,T0,T-', 
            4: 'LL,S,T0', # Direct interations with minimal states
            5: 'LR,S,T0', 
            6: 'RR,S,T-',  
            7: 'LR,T-,T0',
            8: 'LR,T+,T-',
            9: 'LR,T0,T0',
            10: 'LR,T+,T0',  # Rest of states
            11: 'LL,T-,S', 
            12: 'LL,S,T+',
            13: 'LR,S,T+',
            14: 'LR,T0,T+',
            15: 'LR,T+,T+',
            16: 'LL,T0,S', 
            17: 'LL,T+,S',
            18: 'LR,T0,S',
            19: 'LR,T-,S',
            20: 'LR,T+,S',
            21: 'LR,S,S',
            22: 'LR,T-,T+',
            23: 'RR,T-,S', 
            24: 'RR,T0,S',
            25: 'RR,T+,S',
            26: 'RR,S,T0',
            27: 'RR,S,T+',
        }

        self.spinPlus_valleyMinus_correspondence = {
            0: 'LL,S,T-', # (2,0) configurations ordered from Spin T+ to spin T-
            1: 'LL,T0,S',
            2: 'LL,S,T0',
            3: 'LL,T+,S', 
            4: 'LL,T-,S',
            5: 'LL,S,T+',
            6: 'LR,T+,T-', # (1,1) configurations for valley T-
            7: 'LR,S,T-',
            8: 'LR,T0,T-',
            9: 'LR,T-,T-',
            10: 'LR,T+,S', # (1,1) configurations for valley S
            11: 'LR,S,S',
            12: 'LR,T0,S',
            13: 'LR,T-,S', 
            14: 'LR,T+,T0', # (1,1) configurations for valley T0
            15: 'LR,S,T0', 
            16: 'LR,T0,T0',
            17: 'LR,T-,T0',
            18: 'LR,T+,T+', # (1,1) configurations for valley T+
            19: 'LR,S,T+',
            20: 'LR,T0,T+',
            21: 'LR,T-,T+',
            22: 'RR,S,T-', # (0,2) configurations ordered from spin T+ to spin T-
            23: 'RR,T0,S',
            24: 'RR,S,T0', 
            25: 'RR,T+,S', 
            26: 'RR,T-,S',
            27: 'RR,S,T+',
        }

        self.symmetric_antisymmetric_correspondence = {
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

    def _initialize_original_dict(self):
        self.original_correspondence = {}
        for i, state in enumerate(self.FSU.basis):
            a,b = self.FSU.get_occupied_orbitals(state)
            self.original_correspondence[i] = f"|{self.orbital_labels[a]};{self.orbital_labels[b]}>"


    def _build_single_particle_dict(self):
        """
        Constructs the single-particle Hamiltonian dictionary for the double quantum dot system.
        The basis is (LUp+, LDown+, LUp-, LDown-, RUp+, RDown+, RUp-, RDown-)
        """
        b_field = self.parameters['b_field']
        b_parallel = self.parameters['b_parallel']
        mub = self.parameters['mub']
        gs = self.parameters['gs']
        gsLeftFactor = self.parameters['gsLeftFactor']
        gvLeftFactor = self.parameters['gvLeftFactor']
        gv = self.parameters['gv']
        DeltaSO = self.parameters['DeltaSO']
        DeltaKK = self.parameters['DeltaKK']
        t = self.parameters['t']
        t_soc = self.parameters['t_soc']
        Ei = self.parameters['E_i']

        spin_zeeman_splitting_left = 0.5 * gs * mub * b_field * gsLeftFactor
        spin_parallel_splitting_left = 0.5 * gs * mub * b_parallel * gsLeftFactor
        valley_zeeman_splitting_left = 0.5 * gv * mub * b_field * gvLeftFactor
        spin_zeeman_splitting_right = 0.5 * gs * mub * b_field
        spin_parallel_splitting_right = 0.5 * gs * mub * b_parallel
        valley_zeeman_splitting_right = 0.5 * gv * mub * b_field
        kane_mele_splitting = 0.5 * DeltaSO

        # Intradot dynamics
        h = {(0, 0): kane_mele_splitting + valley_zeeman_splitting_left + spin_zeeman_splitting_left,  # LUp+
         (1, 1): - kane_mele_splitting + valley_zeeman_splitting_left - spin_zeeman_splitting_left,  # LDown+
         (2, 2): - kane_mele_splitting - valley_zeeman_splitting_left + spin_zeeman_splitting_left,  # LUp-
         (3, 3): kane_mele_splitting - valley_zeeman_splitting_left - spin_zeeman_splitting_left,  # LDown-
         (4, 4): Ei + kane_mele_splitting + valley_zeeman_splitting_right + spin_zeeman_splitting_right, # RUp+
         (5, 5): Ei - kane_mele_splitting + valley_zeeman_splitting_right - spin_zeeman_splitting_right, # RDown+
         (6, 6): Ei - kane_mele_splitting - valley_zeeman_splitting_right + spin_zeeman_splitting_right, # RUp-
         (7, 7): Ei + kane_mele_splitting - valley_zeeman_splitting_right - spin_zeeman_splitting_right, # RDown-
         (0, 1): spin_parallel_splitting_left,
         (2, 3): spin_parallel_splitting_left,
         (4, 5): spin_parallel_splitting_right,
         (6, 7): spin_parallel_splitting_right,
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
    
    def call_basis_creators(self):
        self.create_original_basis()
        self.create_orbital_symmetry_basis()
        self.create_singlet_triplet_basis()
        self.create_singlet_triplet_reordered_basis()
        self.create_spin_symmetry_basis()
        self.create_valley_symmetry_basis()
        self.create_singlet_triplet_in_spin_basis()
        self.create_spinPlus_valleyMinus_basis()
        self.create_singlet_triplet_minimal_basis()

    def create_original_basis(self):
        vector_basis = []
        for i in range(len(self.FSU.basis)):
            vector_basis.append(np.array([0]*i+[1]*1+[0]*(len(self.FSU.basis)-i-1)))

        self.original_basis = vector_basis
    

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
            self.FSU.create_state_from_occupied_orbitals([1,2]): -1}, # 0: LL,S,T0

            {self.FSU.create_state_from_occupied_orbitals([0,1]): +1}, # 1: LL,S,T

            {self.FSU.create_state_from_occupied_orbitals([2,3]): +1}, # 2: LL,S,T-

            {self.FSU.create_state_from_occupied_orbitals([0,3]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,2]): +1}, # 3: LL,T0,S

            {self.FSU.create_state_from_occupied_orbitals([0,2]): +1}, # 4: LL,T+,S

            {self.FSU.create_state_from_occupied_orbitals([1,3]): +1}, # 5: LL,T-,S
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # (1,1) orbital symmetric configurations

        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([0,7]):+1,
             self.FSU.create_state_from_occupied_orbitals([2,5]):+1,
             self.FSU.create_state_from_occupied_orbitals([1,6]):-1,
             self.FSU.create_state_from_occupied_orbitals([3,4]):-1}, # 6: LR,S,T0

             {self.FSU.create_state_from_occupied_orbitals([0,5]): +1,
              self.FSU.create_state_from_occupied_orbitals([1,4]): -1}, # 7: LR,S,T+

             {self.FSU.create_state_from_occupied_orbitals([2,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([3,6]): -1}, # 8: LR,S,T-

             {self.FSU.create_state_from_occupied_orbitals([0,7]):+1,
             self.FSU.create_state_from_occupied_orbitals([2,5]):-1,
             self.FSU.create_state_from_occupied_orbitals([1,6]):+1,
             self.FSU.create_state_from_occupied_orbitals([3,4]):-1}, # 9: LR,T0,S

             {self.FSU.create_state_from_occupied_orbitals([0,6]): +1,
              self.FSU.create_state_from_occupied_orbitals([2,4]): -1}, # 10: LR,T+,S

             {self.FSU.create_state_from_occupied_orbitals([1,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([3,5]): -1} # 11: LR,T-,S
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
            self.FSU.create_state_from_occupied_orbitals([5,6]): -1}, # 22: RR,S,T0

            {self.FSU.create_state_from_occupied_orbitals([4,5]): +1}, # 23: RR,S,T+

            {self.FSU.create_state_from_occupied_orbitals([6,7]): +1}, # 24: RR,S,T-

            {self.FSU.create_state_from_occupied_orbitals([4,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([5,6]): +1}, # 25: RR,T0,S

            {self.FSU.create_state_from_occupied_orbitals([4,6]): +1}, # 26: RR,T+,S

            {self.FSU.create_state_from_occupied_orbitals([5,7]): +1}, # 27: RR,T-,S
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))


        correspondence = self.singlet_triplet_correspondence

        for i, vec1 in enumerate(list_of_vectors):
            for j, vec2 in enumerate(list_of_vectors):
                overlap = np.vdot(vec1, vec2)
                if i < j and np.abs(overlap) > 1e-6:
                    print(f"No ortogonality between vectors {correspondence[i]} y {correspondence[j]}  in singlet-triplet basis: ⟨{correspondence[i]}|{correspondence[j]}⟩ = {overlap}")

        for i, vec in enumerate(list_of_vectors):
            norm = np.linalg.norm(vec)
            if not np.isclose(norm, 1.0, atol=1e-10):
                print(f"Vector {correspondence[i]}  in singlet-triplet basis is not normalized: ||vec|| = {norm}")

        self.singlet_triplet_basis = list_of_vectors


    def create_spinPlus_valleyMinus_basis(self):

        list_of_vectors = []

        # (2,0) configurations ordered from Spin T+ to spin T-
        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([2,3]): +1}, # 2: LL,S,T-

            {self.FSU.create_state_from_occupied_orbitals([0,3]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,2]): +1}, # 3: LL,T0,S

            {self.FSU.create_state_from_occupied_orbitals([0,3]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,2]): -1}, # 0: LL,S,T0

            {self.FSU.create_state_from_occupied_orbitals([0,2]): +1}, # 4: LL,T+,S

            {self.FSU.create_state_from_occupied_orbitals([1,3]): +1}, # 5: LL,T-,S

            {self.FSU.create_state_from_occupied_orbitals([0,1]): +1}, # 1: LL,S,T+

        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # (1,1) configurations for valley T-


        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([2,6]): +1},  # 18: T+,T-

            {self.FSU.create_state_from_occupied_orbitals([2,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([3,6]): -1}, # 8: LR,S,T-

            {self.FSU.create_state_from_occupied_orbitals([2,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([3,6]): +1},  # 15: T0,T-

            {self.FSU.create_state_from_occupied_orbitals([3,7]): +1}   # 21: T-,T-
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # (1,1) configurations for valley S

        list_of_activations = [

            {self.FSU.create_state_from_occupied_orbitals([0,6]): +1,
              self.FSU.create_state_from_occupied_orbitals([2,4]): -1}, # 10: LR,T+,S

            {self.FSU.create_state_from_occupied_orbitals([0,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([2,5]): -1,
            self.FSU.create_state_from_occupied_orbitals([1,6]): -1,
            self.FSU.create_state_from_occupied_orbitals([3,4]): +1},  # 12: S,S

             {self.FSU.create_state_from_occupied_orbitals([0,7]):+1,
             self.FSU.create_state_from_occupied_orbitals([2,5]):-1,
             self.FSU.create_state_from_occupied_orbitals([1,6]):+1,
             self.FSU.create_state_from_occupied_orbitals([3,4]):-1}, # 9: LR,T0,S

             {self.FSU.create_state_from_occupied_orbitals([1,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([3,5]): -1} # 11: LR,T-,S
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()


        # (1,1) configurations for valley T0

        list_of_activations = [

            {self.FSU.create_state_from_occupied_orbitals([0,6]): +1,
            self.FSU.create_state_from_occupied_orbitals([2,4]): +1},  # 16: T+,T0

            {self.FSU.create_state_from_occupied_orbitals([0,7]):+1,
             self.FSU.create_state_from_occupied_orbitals([2,5]):+1,
             self.FSU.create_state_from_occupied_orbitals([1,6]):-1,
             self.FSU.create_state_from_occupied_orbitals([3,4]):-1}, # 6: LR,S,T0

            {self.FSU.create_state_from_occupied_orbitals([0,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([2,5]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,6]): +1,
            self.FSU.create_state_from_occupied_orbitals([3,4]): +1},  # 13: T0,T0


            {self.FSU.create_state_from_occupied_orbitals([1,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([3,5]): +1},  # 19: T-,T0

        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # (1,1) configurations for valley T+
        list_of_activations = [

            {self.FSU.create_state_from_occupied_orbitals([0,4]): +1},  # 17: T+,T+

            {self.FSU.create_state_from_occupied_orbitals([0,5]): +1,
              self.FSU.create_state_from_occupied_orbitals([1,4]): -1}, # 7: LR,S,T+

            {self.FSU.create_state_from_occupied_orbitals([0,5]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,4]): +1},  # 14: T0,T+

            {self.FSU.create_state_from_occupied_orbitals([1,5]): +1},  # 20: T-,T+
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # (0,2) configurations ordered from spin T+ to spin T-

        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([6,7]): +1}, # 24: RR,S,T-

            {self.FSU.create_state_from_occupied_orbitals([4,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([5,6]): +1}, # 25: RR,T0,S

            {self.FSU.create_state_from_occupied_orbitals([4,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([5,6]): -1}, # 22: RR,S,T0

            {self.FSU.create_state_from_occupied_orbitals([4,6]): +1}, # 26: RR,T+,S

            {self.FSU.create_state_from_occupied_orbitals([5,7]): +1}, # 27: RR,T-,S

            {self.FSU.create_state_from_occupied_orbitals([4,5]): +1}, # 23: RR,S,T+
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))


        correspondence = self.spinPlus_valleyMinus_correspondence

        for i, vec1 in enumerate(list_of_vectors):
            for j, vec2 in enumerate(list_of_vectors):
                overlap = np.vdot(vec1, vec2)
                if i < j and np.abs(overlap) > 1e-6:
                    print(f"No ortogonality between vectors {correspondence[i]} y {correspondence[j]}  in singlet-triplet basis: ⟨{correspondence[i]}|{correspondence[j]}⟩ = {overlap}")

        for i, vec in enumerate(list_of_vectors):
            norm = np.linalg.norm(vec)
            if not np.isclose(norm, 1.0, atol=1e-10):
                print(f"Vector {correspondence[i]}  in singlet-triplet basis is not normalized: ||vec|| = {norm}")

        self.spinPlus_valleyMinus_basis = list_of_vectors

    def create_singlet_triplet_in_spin_basis(self):
        """
        Bais which takes into account the singlet-triplet states of the spin but not in the valley or dot DOF
        """
        list_of_vectors = []

        # (2,0) configurations
        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([2,3]): +1}, # 0: LL,S,--

            {self.FSU.create_state_from_occupied_orbitals([1,3]): +1}, # 1: LL,T-,+-

            {self.FSU.create_state_from_occupied_orbitals([0,3]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,2]): -1}, # 2: LL,S,+-

            {self.FSU.create_state_from_occupied_orbitals([0,3]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,2]): +1}, # 3: LL,T0,+-

            {self.FSU.create_state_from_occupied_orbitals([0,2]): +1}, # 4: LL,T+,+-

            {self.FSU.create_state_from_occupied_orbitals([0,1]): +1} # 5: LL,S,++
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # (1,1) -- valley states

        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([3,7]):+1}, # 6: LR,T-,--

             {self.FSU.create_state_from_occupied_orbitals([2,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([3,6]): -1}, # 7: LR,S,--

             {self.FSU.create_state_from_occupied_orbitals([2,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([3,6]): +1}, # 8: LR,T0,--

             {self.FSU.create_state_from_occupied_orbitals([2,6]):+1} # 9: LR,T+,--
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # (1,1) +- or -+ valley states

        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([3,5]):+1}, # 10: LR,T-,-+

             {self.FSU.create_state_from_occupied_orbitals([1,7]): +1}, # 11: LR,T-,+-

             {self.FSU.create_state_from_occupied_orbitals([2,5]): +1,
              self.FSU.create_state_from_occupied_orbitals([3,4]): -1}, # 12: LR,S,-+

             {self.FSU.create_state_from_occupied_orbitals([0,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([1,6]): -1}, # 13: LR,S,+-

             {self.FSU.create_state_from_occupied_orbitals([2,5]): +1,
              self.FSU.create_state_from_occupied_orbitals([3,4]): +1}, # 14: LR,T0,-+

             {self.FSU.create_state_from_occupied_orbitals([0,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([1,6]): +1}, # 15: LR,T0,+-

             {self.FSU.create_state_from_occupied_orbitals([2,4]): +1}, # 16: LR,T+,-+

             {self.FSU.create_state_from_occupied_orbitals([0,6]):+1} # 17: LR,T+,+-
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # (1,1) ++ valley states

        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([1,5]):+1}, # 18: LR,T-,++

             {self.FSU.create_state_from_occupied_orbitals([0,5]): +1,
              self.FSU.create_state_from_occupied_orbitals([1,4]): -1}, # 19: LR,S,++

             {self.FSU.create_state_from_occupied_orbitals([0,5]): +1,
              self.FSU.create_state_from_occupied_orbitals([1,4]): +1}, # 20: LR,T0,++

             {self.FSU.create_state_from_occupied_orbitals([0,4]):+1} # 21: LR,T+,++
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # (0,2) configurations

        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([6,7]): +1}, # 22: RR,S,--

            {self.FSU.create_state_from_occupied_orbitals([5,7]): +1}, # 23: RR,T-,+-

            {self.FSU.create_state_from_occupied_orbitals([4,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([5,6]): -1}, # 24: RR,S,+-

            {self.FSU.create_state_from_occupied_orbitals([4,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([5,6]): +1}, # 25: RR,T0,+-

            {self.FSU.create_state_from_occupied_orbitals([4,6]): +1}, # 26: RR,T+,+-

            {self.FSU.create_state_from_occupied_orbitals([4,5]): +1} # 27: RR,S,++
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))


        correspondence = self.singlet_triplet_in_spin_correspondence

        for i, vec1 in enumerate(list_of_vectors):
            for j, vec2 in enumerate(list_of_vectors):
                overlap = np.vdot(vec1, vec2)
                if i < j and np.abs(overlap) > 1e-6:
                    print(f"No ortogonality between vectors {correspondence[i]} y {correspondence[j]}  in singlet-triplet basis: ⟨{correspondence[i]}|{correspondence[j]}⟩ = {overlap}")

        for i, vec in enumerate(list_of_vectors):
            norm = np.linalg.norm(vec)
            if not np.isclose(norm, 1.0, atol=1e-10):
                print(f"Vector {correspondence[i]}  in singlet-triplet basis is not normalized: ||vec|| = {norm}")

        self.singlet_triplet_in_spin_basis = list_of_vectors

    def create_singlet_triplet_reordered_basis(self):
        """
        Same as before but with elements reordered to trace out all high energetic eigenstates
        """

        list_of_vectors = []

        # less energetic configurations
        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([2,6]): +1},  # LR,T+,T-

            {self.FSU.create_state_from_occupied_orbitals([2,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([3,6]): -1}, # LR,S,T-

            {self.FSU.create_state_from_occupied_orbitals([2,3]): +1}, # LL,S,T-

            {self.FSU.create_state_from_occupied_orbitals([2,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([3,6]): +1},  # LR,T0,T-

            {self.FSU.create_state_from_occupied_orbitals([3,7]): +1},   # LR,T-,T-
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # Interactions

        list_of_activations = [

            {self.FSU.create_state_from_occupied_orbitals([0,6]): +1,
              self.FSU.create_state_from_occupied_orbitals([2,4]): +1},  #  LR,T+,T0

             {self.FSU.create_state_from_occupied_orbitals([0,3]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,2]): -1}, # LL,S,T0

            {self.FSU.create_state_from_occupied_orbitals([0,7]):+1,
             self.FSU.create_state_from_occupied_orbitals([2,5]):+1,
             self.FSU.create_state_from_occupied_orbitals([1,6]):-1,
             self.FSU.create_state_from_occupied_orbitals([3,4]):-1}, # LR,S,T0

            {self.FSU.create_state_from_occupied_orbitals([6,7]): +1}, # RR,S,T-


            {self.FSU.create_state_from_occupied_orbitals([0,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([2,5]): +1,
              self.FSU.create_state_from_occupied_orbitals([1,6]): +1,
              self.FSU.create_state_from_occupied_orbitals([3,4]): +1},  # LR,T0,T0

            {self.FSU.create_state_from_occupied_orbitals([1,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([3,5]): +1},  # LR,T-,T0

        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # Rest

        list_of_activations = [

            {self.FSU.create_state_from_occupied_orbitals([1,3]): +1}, # LL,T-,S

            {self.FSU.create_state_from_occupied_orbitals([0,1]): +1}, # LL,S,T+

            {self.FSU.create_state_from_occupied_orbitals([0,5]): +1,
              self.FSU.create_state_from_occupied_orbitals([1,4]): -1}, # LR,S,T+

            {self.FSU.create_state_from_occupied_orbitals([0,5]): +1,
              self.FSU.create_state_from_occupied_orbitals([1,4]): +1},  #  LR,T0,T+

            {self.FSU.create_state_from_occupied_orbitals([0,4]): +1},  # LR,T+,T+

            {self.FSU.create_state_from_occupied_orbitals([0,3]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,2]): +1}, # LL,T0,S

            {self.FSU.create_state_from_occupied_orbitals([0,2]): +1}, # LL,T+,S

             {self.FSU.create_state_from_occupied_orbitals([0,7]):+1,
             self.FSU.create_state_from_occupied_orbitals([2,5]):-1,
             self.FSU.create_state_from_occupied_orbitals([1,6]):+1,
             self.FSU.create_state_from_occupied_orbitals([3,4]):-1}, # LR,T0,S

             {self.FSU.create_state_from_occupied_orbitals([1,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([3,5]): -1}, # LR,T-,S

             {self.FSU.create_state_from_occupied_orbitals([0,6]): +1,
              self.FSU.create_state_from_occupied_orbitals([2,4]): -1}, # LR,T+,S

             {self.FSU.create_state_from_occupied_orbitals([0,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([2,5]): -1,
              self.FSU.create_state_from_occupied_orbitals([1,6]): -1,
              self.FSU.create_state_from_occupied_orbitals([3,4]): +1},  # S,S

            {self.FSU.create_state_from_occupied_orbitals([1,5]): +1},  # T-,T+

            {self.FSU.create_state_from_occupied_orbitals([5,7]): +1}, # RR,T-,S

            {self.FSU.create_state_from_occupied_orbitals([4,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([5,6]): +1}, # RR,T0,S

            {self.FSU.create_state_from_occupied_orbitals([4,6]): +1}, # RR,T+,S

            {self.FSU.create_state_from_occupied_orbitals([4,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([5,6]): -1}, # RR,S,T0

            {self.FSU.create_state_from_occupied_orbitals([4,5]): +1}, # RR,S,T+
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()


        correspondence = self.singlet_triplet_reordered_correspondence

        for i, vec1 in enumerate(list_of_vectors):
            for j, vec2 in enumerate(list_of_vectors):
                overlap = np.vdot(vec1, vec2)
                if i < j and np.abs(overlap) > 1e-6:
                    print(f"No ortogonality between vectors {correspondence[i]} y {correspondence[j]}  in singlet-triplet basis: ⟨{correspondence[i]}|{correspondence[j]}⟩ = {overlap}")

        for i, vec in enumerate(list_of_vectors):
            norm = np.linalg.norm(vec)
            if not np.isclose(norm, 1.0, atol=1e-10):
                print(f"Vector {correspondence[i]}  in singlet-triplet basis is not normalized: ||vec|| = {norm}")

        self.singlet_triplet_reordered_basis = list_of_vectors


    def create_singlet_triplet_minimal_basis(self):
        """
        Same as before but with elements reordered to trace out all high energetic eigenstates
        """

        list_of_vectors = []

        # less energetic configurations
        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([3,7]): +1},   # LR,T-,T-

            {self.FSU.create_state_from_occupied_orbitals([2,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([3,6]): -1}, # LR,S,T-

            {self.FSU.create_state_from_occupied_orbitals([2,3]): +1}, # LL,S,T-

            {self.FSU.create_state_from_occupied_orbitals([2,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([3,6]): +1},  # LR,T0,T-
            
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # Interactions

        list_of_activations = [

            

             {self.FSU.create_state_from_occupied_orbitals([0,3]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,2]): -1}, # LL,S,T0

            {self.FSU.create_state_from_occupied_orbitals([0,7]):+1,
             self.FSU.create_state_from_occupied_orbitals([2,5]):+1,
             self.FSU.create_state_from_occupied_orbitals([1,6]):-1,
             self.FSU.create_state_from_occupied_orbitals([3,4]):-1}, # LR,S,T0

            {self.FSU.create_state_from_occupied_orbitals([6,7]): +1}, # RR,S,T-


            {self.FSU.create_state_from_occupied_orbitals([1,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([3,5]): +1},  # LR,T-,T0

            {self.FSU.create_state_from_occupied_orbitals([2,6]): +1},  # LR,T+,T-

            {self.FSU.create_state_from_occupied_orbitals([0,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([2,5]): +1,
              self.FSU.create_state_from_occupied_orbitals([1,6]): +1,
              self.FSU.create_state_from_occupied_orbitals([3,4]): +1},  # LR,T0,T0

        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # Rest

        list_of_activations = [

            {self.FSU.create_state_from_occupied_orbitals([0,6]): +1,
              self.FSU.create_state_from_occupied_orbitals([2,4]): +1},  #  LR,T+,T0

            {self.FSU.create_state_from_occupied_orbitals([1,3]): +1}, # LL,T-,S

            {self.FSU.create_state_from_occupied_orbitals([0,1]): +1}, # LL,S,T+

            {self.FSU.create_state_from_occupied_orbitals([0,5]): +1,
              self.FSU.create_state_from_occupied_orbitals([1,4]): -1}, # LR,S,T+

            {self.FSU.create_state_from_occupied_orbitals([0,5]): +1,
              self.FSU.create_state_from_occupied_orbitals([1,4]): +1},  #  LR,T0,T+

            {self.FSU.create_state_from_occupied_orbitals([0,4]): +1},  # LR,T+,T+

            {self.FSU.create_state_from_occupied_orbitals([0,3]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,2]): +1}, # LL,T0,S

            {self.FSU.create_state_from_occupied_orbitals([0,2]): +1}, # LL,T+,S

             {self.FSU.create_state_from_occupied_orbitals([0,7]):+1,
             self.FSU.create_state_from_occupied_orbitals([2,5]):-1,
             self.FSU.create_state_from_occupied_orbitals([1,6]):+1,
             self.FSU.create_state_from_occupied_orbitals([3,4]):-1}, # LR,T0,S

             {self.FSU.create_state_from_occupied_orbitals([1,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([3,5]): -1}, # LR,T-,S

             {self.FSU.create_state_from_occupied_orbitals([0,6]): +1,
              self.FSU.create_state_from_occupied_orbitals([2,4]): -1}, # LR,T+,S

             {self.FSU.create_state_from_occupied_orbitals([0,7]): +1,
              self.FSU.create_state_from_occupied_orbitals([2,5]): -1,
              self.FSU.create_state_from_occupied_orbitals([1,6]): -1,
              self.FSU.create_state_from_occupied_orbitals([3,4]): +1},  # S,S

            {self.FSU.create_state_from_occupied_orbitals([1,5]): +1},  # T-,T+

            {self.FSU.create_state_from_occupied_orbitals([5,7]): +1}, # RR,T-,S

            {self.FSU.create_state_from_occupied_orbitals([4,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([5,6]): +1}, # RR,T0,S

            {self.FSU.create_state_from_occupied_orbitals([4,6]): +1}, # RR,T+,S

            {self.FSU.create_state_from_occupied_orbitals([4,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([5,6]): -1}, # RR,S,T0

            {self.FSU.create_state_from_occupied_orbitals([4,5]): +1}, # RR,S,T+
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()


        correspondence = self.singlet_triplet_reordered_correspondence

        for i, vec1 in enumerate(list_of_vectors):
            for j, vec2 in enumerate(list_of_vectors):
                overlap = np.vdot(vec1, vec2)
                if i < j and np.abs(overlap) > 1e-6:
                    print(f"No ortogonality between vectors {correspondence[i]} y {correspondence[j]}  in singlet-triplet basis: ⟨{correspondence[i]}|{correspondence[j]}⟩ = {overlap}")

        for i, vec in enumerate(list_of_vectors):
            norm = np.linalg.norm(vec)
            if not np.isclose(norm, 1.0, atol=1e-10):
                print(f"Vector {correspondence[i]}  in singlet-triplet basis is not normalized: ||vec|| = {norm}")

        self.singlet_triplet_minimal_basis = list_of_vectors

    def create_orbital_symmetry_basis(self):
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
             self.FSU.create_state_from_occupied_orbitals([4,7]): -self.b*self.a2} #S6
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # orbital antisymmetric configurations
        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([0,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([3,4]): +1}, #AS1

            {self.FSU.create_state_from_occupied_orbitals([3,7]): +1}, #AS2

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


        correspondence = self.symmetric_antisymmetric_correspondence

        for i, vec1 in enumerate(list_of_vectors.copy()):
            for j, vec2 in enumerate(list_of_vectors.copy()):
                overlap = np.vdot(vec1, vec2)
                if i < j and np.abs(overlap) > 1e-6:
                    print(f"No ortogonality between vectors {correspondence[i]} y {correspondence[j]}  in orbital basis: ⟨{correspondence[i]}|{correspondence[j]}⟩ = {overlap}")

        for i, vec in enumerate(list_of_vectors.copy()):
            norm = np.linalg.norm(vec)
            if not np.isclose(norm, 1.0, atol=1e-10):
                print(f"Vector {correspondence[i]}  in orbital basis is not normalized: ||vec|| = {norm}")

        self.orbital_symmetry_basis = list_of_vectors


    def create_spin_symmetry_basis(self):
        """
        There are 16 basis elements:
        - 10 for antysimmetric spin part
        - 6 for symmetric spin part
        
        This method takes the basis expressed as integers numbers which represents bit determinants exprresing c_i^dag c_j^dag |0>
        with i,j in 1,..,8 as in orbital_labels and the returns 16 vectors as coeficients expressed in this basis (28 elements per vector) which forms
        the symmetic-antisymmetric basis in the spin.
        """

        self.compute_some_characteristic_properties()
        list_of_vectors = []

        # spin symmetric configurations

        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([0,7]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([3,4]): -1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([1,6]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([2,5]): -1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([0,6]): 1.0,
             self.FSU.create_state_from_occupied_orbitals([1,7]): 1.0,
             self.FSU.create_state_from_occupied_orbitals([2,4]): -1.0,
             self.FSU.create_state_from_occupied_orbitals([3,5]): -1.0}, #S1

             {self.FSU.create_state_from_occupied_orbitals([4,6]): 1.0,
             self.FSU.create_state_from_occupied_orbitals([4,7]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([5,6]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([5,7]): 1.0}, #S2

             {self.FSU.create_state_from_occupied_orbitals([0,2]): 1.0,
             self.FSU.create_state_from_occupied_orbitals([0,3]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([1,2]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([1,3]): 1.0}, #S3

             {self.FSU.create_state_from_occupied_orbitals([0,4]): 1.0,
             self.FSU.create_state_from_occupied_orbitals([0,5]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([1,4]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([1,5]): 1.0}, #S4

             {self.FSU.create_state_from_occupied_orbitals([2,6]): 1.0,
             self.FSU.create_state_from_occupied_orbitals([2,7]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([3,6]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([3,7]): 1.0}, #S5

             {self.FSU.create_state_from_occupied_orbitals([0,7]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([3,4]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([1,6]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([2,5]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([0,6]): 1.0,
             self.FSU.create_state_from_occupied_orbitals([1,7]): 1.0,
             self.FSU.create_state_from_occupied_orbitals([2,4]): 1.0,
             self.FSU.create_state_from_occupied_orbitals([3,5]): 1.0}, #S6
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # spin antisymmetric configurations
        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([0,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,6]): -1}, 

            {self.FSU.create_state_from_occupied_orbitals([6,7]): +1}, 

            {self.FSU.create_state_from_occupied_orbitals([0,1]): +1},  

            {self.FSU.create_state_from_occupied_orbitals([2,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([3,6]): -1},  

            {self.FSU.create_state_from_occupied_orbitals([0,3]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,2]): -1}, # AS6

            {self.FSU.create_state_from_occupied_orbitals([4,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([5,6]): -1},   

            {self.FSU.create_state_from_occupied_orbitals([0,5]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,4]): -1},   

            {self.FSU.create_state_from_occupied_orbitals([2,5]): +1,
            self.FSU.create_state_from_occupied_orbitals([3,4]): -1}, 

            {self.FSU.create_state_from_occupied_orbitals([2,3]): +1},  

            {self.FSU.create_state_from_occupied_orbitals([4,5]): +1},  
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))


        correspondence = self.symmetric_antisymmetric_correspondence

        for i, vec1 in enumerate(list_of_vectors.copy()):
            for j, vec2 in enumerate(list_of_vectors.copy()):
                overlap = np.vdot(vec1, vec2)
                if i < j and np.abs(overlap) > 1e-6:
                    print(f"No ortogonality between vectors {correspondence[i]} y {correspondence[j]} in spin basis: ⟨{correspondence[i]}|{correspondence[j]}⟩ = {overlap}")

        for i, vec in enumerate(list_of_vectors.copy()):
            norm = np.linalg.norm(vec)
            if not np.isclose(norm, 1.0, atol=1e-10):
                print(f"Vector {correspondence[i]}  in spin basis is not normalized: ||vec|| = {norm}")

        self.spin_symmetry_basis = list_of_vectors


    def create_valley_symmetry_basis(self):
        """
        There are 16 basis elements:
        - 10 for antysimmetric valley part
        - 6 for symmetric valley part
        
        This method takes the basis expressed as integers numbers which represents bit determinants exprresing c_i^dag c_j^dag |0>
        with i,j in 1,..,8 as in orbital_labels and the returns 16 vectors as coeficients expressed in this basis (28 elements per vector) which forms
        the symmetic-antisymmetric basis in the valley dof.
        """

        self.compute_some_characteristic_properties()
        list_of_vectors = []

        # spin symmetric configurations

        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([0,7]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([3,4]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([1,6]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([2,5]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([0,5]): 1.0,
             self.FSU.create_state_from_occupied_orbitals([1,4]): 1.0,
             self.FSU.create_state_from_occupied_orbitals([2,7]): 1.0,
             self.FSU.create_state_from_occupied_orbitals([3,6]): 1.0}, #S1

             {self.FSU.create_state_from_occupied_orbitals([1,5]): 1.0,
             self.FSU.create_state_from_occupied_orbitals([1,7]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([3,5]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([3,7]): 1.0}, #S2

             {self.FSU.create_state_from_occupied_orbitals([0,4]): 1.0,
             self.FSU.create_state_from_occupied_orbitals([2,4]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([0,6]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([2,6]): 1.0}, #S3

             {self.FSU.create_state_from_occupied_orbitals([0,1]): -1.0,
             self.FSU.create_state_from_occupied_orbitals([1,2]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([0,3]): -1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([2,3]): -1.0}, #S4

             {self.FSU.create_state_from_occupied_orbitals([4,5]): 1.0,
             self.FSU.create_state_from_occupied_orbitals([4,7]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([5,6]): -1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([6,7]): 1.0}, #S5

             {self.FSU.create_state_from_occupied_orbitals([0,7]): -1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([3,4]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([1,6]): 1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([2,5]): -1.0/np.sqrt(2),
             self.FSU.create_state_from_occupied_orbitals([0,5]): -1.0,
             self.FSU.create_state_from_occupied_orbitals([2,7]): -1.0,
             self.FSU.create_state_from_occupied_orbitals([1,4]): 1.0,
             self.FSU.create_state_from_occupied_orbitals([3,6]): 1.0}, #S6
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        list_of_activations.clear()

        # spin antisymmetric configurations
        list_of_activations = [
            {self.FSU.create_state_from_occupied_orbitals([0,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([2,5]): -1}, 

            {self.FSU.create_state_from_occupied_orbitals([5,7]): +1}, 

            {self.FSU.create_state_from_occupied_orbitals([0,2]): +1},  

            {self.FSU.create_state_from_occupied_orbitals([4,7]): +1,
            self.FSU.create_state_from_occupied_orbitals([5,6]): +1},  

            {self.FSU.create_state_from_occupied_orbitals([2,4]): -1,
            self.FSU.create_state_from_occupied_orbitals([0,6]): 1}, 

            {self.FSU.create_state_from_occupied_orbitals([3,5]): -1,
            self.FSU.create_state_from_occupied_orbitals([1,7]): 1}, # AS6

            {self.FSU.create_state_from_occupied_orbitals([0,3]): +1,
            self.FSU.create_state_from_occupied_orbitals([1,2]): 1},   

            {self.FSU.create_state_from_occupied_orbitals([1,6]): +1,
            self.FSU.create_state_from_occupied_orbitals([3,4]): -1},   

            {self.FSU.create_state_from_occupied_orbitals([4,6]): +1},  

            {self.FSU.create_state_from_occupied_orbitals([1,3]): +1},  
        ]

        for activation in list_of_activations:
            list_of_vectors.append(self.FSU.create_normalized_vector(activation))

        correspondence = self.symmetric_antisymmetric_correspondence

        for i, vec1 in enumerate(list_of_vectors):
            for j, vec2 in enumerate(list_of_vectors):
                overlap = np.vdot(vec1, vec2)
                if i < j and np.abs(overlap) > 1e-6:
                    print(f"No ortogonality between vectors {correspondence[i]} y {correspondence[j]}  in valley basis: ⟨{correspondence[i]}|{correspondence[j]}⟩ = {overlap}")

        for i, vec in enumerate(list_of_vectors):
            norm = np.linalg.norm(vec)
            if not np.isclose(norm, 1.0, atol=1e-10):
                print(f"Vector {correspondence[i]}  in valley basis is not normalized: ||vec|| = {norm}")

        self.valley_symmetry_basis = list_of_vectors

    def calculate_eigenvalues_and_eigenvectors(self, parameters_to_change=None):
        Hmatrix = self.obtain_hamiltonian_determinant_basis(parameters_to_change=parameters_to_change)
        eigval, eigv = eigh(Hmatrix)

        return eigval, eigv
    
    def obtain_hamiltonian_determinant_basis(self, parameters_to_change=None):

        if parameters_to_change is not None:
            self.parameters.update(parameters_to_change)

        h = self._build_single_particle_dict()
        V = self._build_interaction_dict()
        H = ManyBodyHamiltonian(self.norb, h, V)
        H.build(self.n_elec)

        return H.matrix
    
    def project_hamiltonian(self, list_of_vectors, parameters_to_change=None, alternative_operator = None):
        Umatrix = np.array(list_of_vectors).T  # shape: (dimFull, nProj)
        assert np.allclose(Umatrix.conj().T @ Umatrix, np.eye(Umatrix.shape[1]))

        Hmatrix = alternative_operator

        if Hmatrix is None:
            Hmatrix = self.obtain_hamiltonian_determinant_basis(parameters_to_change=parameters_to_change)

        return Umatrix.conj().T @ Hmatrix @ Umatrix

    

    def diagnoseProjectionQuality(self, listOfVectors, parametersToChange=None, fidelityThreshold=0.98, energyTolerance=1e-3, verbose=True):
        """
        Evaluate whether projecting the Hamiltonian onto a reduced basis is a good approximation.

        Parameters:
        - listOfVectors: list or array of orthonormal vectors defining the projected subspace (shape: nProj × dimFull)
        - parametersToChange: optional dictionary of parameters to update before building the Hamiltonian
        - fidelityThreshold: minimum fidelity required to consider the projection valid
        - energyTolerance: maximum acceptable deviation in energy between full and projected spectra
        - verbose: whether to print per-state fidelities and energy differences

        Returns:
        - Dictionary with fidelity list, energy differences, and flags indicating quality
        """

        # Get the full Hamiltonian in determinant basis
        Hfull = self.obtain_hamiltonian_determinant_basis(parameters_to_change=parametersToChange)
        eigvalsFull, eigvecsFull = np.linalg.eigh(Hfull)

        # Projection matrix P with orthonormal rows
        P = np.array(listOfVectors).T  # shape: (dimFull, nProj)
        assert np.allclose(P.conj().T @ P, np.eye(P.shape[1])), "Projected basis is not orthonormal"


        # Projected Hamiltonian
        Hproj = P.conj().T @ Hfull @ P
        eigvalsProj, eigvecsProj = np.linalg.eigh(Hproj)

        nProj = len(listOfVectors)
        fidelityList = []
        energyDiffList = []

        for i in range(nProj):
            psiFull = eigvecsFull[:, i]
            psiProj = P @ (P.conj().T @ psiFull)
            fidelity = np.abs(np.vdot(psiFull, psiProj))**2
            deltaE = np.abs(eigvalsFull[i] - eigvalsProj[i])
            fidelityList.append(fidelity)
            energyDiffList.append(deltaE)

            if verbose:
                print(f"State {i}: Fidelity = {fidelity:.6f}, ΔE = {deltaE:.6e}")

        # Global diagnosis
        fidelitiesOk = all(f > fidelityThreshold for f in fidelityList)
        energiesOk = all(dE < energyTolerance for dE in energyDiffList)

        print("\nProjection diagnostic:")
        if fidelitiesOk and energiesOk:
            print("Projection is safe: high fidelities and accurate spectrum reproduction.")
        elif not fidelitiesOk and energiesOk:
            print("Warning: low fidelities despite accurate energies. Projection may miss state components.")
        elif fidelitiesOk and not energiesOk:
            print("Warning: high fidelities but energy mismatch. Possible degeneracies or level mixing.")
        else:
            print("Projection is not reliable: significant loss of information.")

        return {
            "fidelities": fidelityList,
            "energyDiffs": energyDiffList,
            "fidelitiesOk": fidelitiesOk,
            "energiesOk": energiesOk
        }

    

    def compute_some_characteristic_properties(self):
        U0 = self.parameters[DQDParameters.U0.value]
        U1 = self.parameters[DQDParameters.U1.value]
        t = self.parameters[DQDParameters.T.value]
        J = self.parameters[DQDParameters.J.value]
        g_ortho = self.parameters[DQDParameters.G_ORTHO.value]
        g_zz = self.parameters[DQDParameters.G_ZZ.value]
        g_z0 = self.parameters[DQDParameters.G_Z0.value]
        g_0z = self.parameters[DQDParameters.G_0Z.value]
        DeltaSO = self.parameters[DQDParameters.DELTA_SO.value]


        self.a1 = 1.0/np.sqrt(1+16*t**2/(U0-U1+np.sqrt(16*t**2+(U0-U1)**2))**2)
        self.a2 = 1.0/(self.a1*np.sqrt(4+(U0-U1)**2/(4*t**2))*np.sqrt(2))
        self.alpha = 1.0/(3+ (U0-U1)**2/(4*t**2))
        self.DeltaOrb = - (U0-U1)/ 2.0 + 0.5*np.sqrt((U0-U1)**2 + 16*t**2)
        self.DiffOrb = self.DeltaOrb - J *self.alpha* g_zz
        self.DiffIntraOrb = 4*DeltaSO + 8 * J*self.alpha * g_ortho
        self.b = self.alpha * g_ortho * J / DeltaSO if abs(DeltaSO) > 1e-5 else 1000
        self.C = np.sqrt(1+self.b**2)
        self.Jeff = self.DeltaOrb - self.alpha * 2* J *(g_z0 + g_0z + 0.5*g_zz)


    def buildSzMatrix(self, dof: str) -> np.ndarray:
        dof_patterns = {
            "spin": {'Up': 0.5, 'Down': -0.5},
            "valley": {'+': 0.5, '-': -0.5},
            "dot": {'L': 0.5, 'R': -0.5}
        }
        
        if dof not in dof_patterns:
            raise ValueError("The only dof allowed are 'spin', 'valley' or 'dot'")
        
        dim = len(self.FSU.basis)
        szMatrix = np.zeros((dim, dim))
        patterns = dof_patterns[dof]
        
        for i, det in enumerate(self.FSU.basis):
            occupied = self.FSU.get_occupied_orbitals(det)
            sz = 0.0
            for orb in occupied:
                label = self.orbital_labels[orb]
                for pattern, value in patterns.items():
                    if pattern in label:
                        sz += value
                        break
            szMatrix[i, i] = sz
        
        return szMatrix

    def buildSpinLadderMatrix(self, ladderType: str, dof: str) -> np.ndarray:
        dof_patterns = {
            "spin": ('Down', 'Up'),
            "valley": ('-', '+'),
            "dot": ('R', 'L')
        }
        
        if dof not in dof_patterns:
            raise ValueError("The only dof allowed are 'spin', 'valley' or 'dot'")
        if ladderType not in ('plus', 'minus'):
            raise ValueError("ladderType must be 'plus' or 'minus'")
        
        dim = len(self.FSU.basis)
        matrix = np.zeros((dim, dim))
        old_pattern, new_pattern = dof_patterns[dof]
        
        for i, det in enumerate(self.FSU.basis):
            for orb in range(self.norb):
                label = self.orbital_labels[orb]
                if old_pattern in label:
                    partnerLabel = label.replace(old_pattern, new_pattern)
                    partnerOrb = next((j for j, lbl in self.orbital_labels.items() 
                                    if lbl == partnerLabel), None)
                    if partnerOrb is None:
                        continue

                    annihilateOrb, createOrb = (orb, partnerOrb) if ladderType == 'plus' else (partnerOrb, orb)

                    newState, phase1 = self.FSU.apply_annihilation(det, annihilateOrb)
                    if newState is not None:
                        finalState, phase2 = self.FSU.apply_creation(newState, createOrb)
                        if finalState is not None and finalState in self.FSU.basis:
                            j = self.FSU.basis.index(finalState)
                            matrix[j, i] += phase1 * phase2
        
        return matrix
    

    def buildS2Matrix(self, dof: str) -> np.ndarray:
        sz = self.buildSzMatrix(dof)
        sp = self.buildSpinLadderMatrix('plus', dof)
        sm = self.buildSpinLadderMatrix('minus', dof)
        
        s2 = sz @ sz + 0.5 * (sp @ sm + sm @ sp)
        return s2
    
    def expectationValue(self, operatorMatrix: np.ndarray, stateVector: np.ndarray) -> float:
        return np.real(np.vdot(stateVector, operatorMatrix @ stateVector))






        