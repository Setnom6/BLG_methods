from many_body_builder import ManyBodyHamiltonian, get_occupied_orbitals
import numpy as np
from scipy.linalg import eigh

import matplotlib.pyplot as plt

from enum import Enum


class DQDParameters(Enum):
    B_FIELD = 'b_field'
    B_PARALLEL = 'b_parallel'
    MUB = 'mub'
    GS = 'gs'
    GV = 'gv'
    DELTA_SO = 'DeltaSO'
    DELTA_KK = 'DeltaKK'
    DELTA_ORB = 'DeltaOrb'
    E_I = 'E_i'
    T11 = 't11'
    T12 = 't12'
    T22 = 't22'
    T_SOC11 = 't_soc11'
    T_SOC12 = 't_soc12'
    T_SOC22 = 't_soc22'
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
            'DeltaOrb': 10.0, # Orbital splitting in meV
            'E_i': 0.0, # Single-particle energy level in eV expressed as a difference between the two dots
            't11': 0.04, # Spin-conserving hopping in meV between orbitals 1 and 1 in each dot
            't12': 0.04, # Spin-conserving hopping in meV between orbitals 1 and 2 in each dot. We assume that hopping between same dot and different orbital is not allowed
            't22': 0.04, # Spin-conserving hopping in meV between orbitals 2 and 2 in each dot
            't_soc11': 0.0, # Spin-flip hopping in meV due to extrinsic spin-orbit coupling between orbitals 1 and 1 in each dot
            't_soc12': 0.0, # Spin-flip hopping in meV due to extrinsic spin-orbit coupling between orbitals 1 and 2 in each dot
            't_soc22': 0.0, # Spin-flip hopping in meV due to extrinsic spin-orbit coupling between orbitals 2 and 2 in each dot
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
            0: 'L1Up+',
            1: 'L1Down+',
            2: 'L1Up-',
            3: 'L1Down-',
            4: 'L2Up+',
            5: 'L2Down+',
            6: 'L2Up-',
            7: 'L2Down-', 
            8: 'R1Up+',
            9: 'R1Down+',
            10: 'R1Up-',
            11: 'R1Down-',
            12: 'R2Up+',
            13: 'R2Down+',
            14: 'R2Up-',
            15: 'R2Down-', 
        }

        self.labels_to_indices = {v: k for k, v in self.orbital_labels.items()}

        if parameters_to_change is not None:
            for key, value in parameters_to_change.items():
                if key in self.parameters:
                    self.parameters[key] = value
                else:
                    raise ValueError(f"Parameter {key} is not recognized.")
                

        self.N = 16 # Number of sites in the system (L1Up+, L1Down+, L1Up-, L1Down-, R1Up+, R1Down+, R1Up-, R1Down-, L2Up+, L2Down+, L2Up-, L2Down-, R2Up+, R2Down+, R2Up-, R2Down-)
        self.numParticles = 2 # Number of particles in the system (two electrons)
        h = self._build_single_particle_dict()
        V = self._build_interaction_dict()
        H = ManyBodyHamiltonian(self.N, h, V)
        self.basis = H.generate_basis(self.numParticles)  # Generate the basis for two electrons

    def _build_single_particle_dict(self):
        """
        Constructs the single-particle Hamiltonian dictionary for the double quantum dot system.
        Basis: (L1Up+, L1Down+, L1Up-, L1Down-, L2Up+, L2Down+, L2Up-, L2Down-, R1Up+, R1Down+, R1Up-, R1Down-, R2Up+, R2Down+, R2Up-, R2Down-)
        """
        p = self.parameters
        sz = 0.5 * p['gs'] * p['mub'] * p['b_field']  # spin Zeeman
        sp = 0.5 * p['gs'] * p['mub'] * p['b_parallel']  # spin parallel
        vz = 0.5 * p['gv'] * p['mub'] * p['b_field']  # valley Zeeman
        km = 0.5 * p['DeltaSO']  # Kane-Mele

        # Helper function to generate intradot terms
        def intradot_terms(dot_offset, energy_offset):
            return {
                (0+dot_offset, 0+dot_offset): energy_offset + km + vz + sz,
                (1+dot_offset, 1+dot_offset): energy_offset - km + vz - sz,
                (2+dot_offset, 2+dot_offset): energy_offset - km - vz + sz,
                (3+dot_offset, 3+dot_offset): energy_offset + km - vz - sz,
                (0+dot_offset, 1+dot_offset): sp,
                (2+dot_offset, 3+dot_offset): sp,
                (0+dot_offset, 2+dot_offset): p['DeltaKK'],
                (1+dot_offset, 3+dot_offset): p['DeltaKK']
            }

        # Build Hamiltonian
        h = {}
        h.update(intradot_terms(0, 0))  # L1
        h.update(intradot_terms(4, p['DeltaOrb']))  # L2
        h.update(intradot_terms(8, p['E_i']))  # R1
        h.update(intradot_terms(12, p['E_i'] + p['DeltaOrb']))  # R2

        # Interdot tunneling (direct and SOC)
        def add_interdot_tunneling(h, src_dots, t, t_soc):
            for i in range(4):  # 4 spin-valley combinations
                h[(src_dots[0]+i, src_dots[1]+i)] = t
                h[(src_dots[0]+i, src_dots[1]+(i^1))] = 1j * t_soc * (-1 if i%2 else 1)
        
        # Add all interdot connections
        interdot_pairs = [
            (0, 8, p['t11'], p['t_soc11']),  # L1-R1
            (4, 12, p['t22'], p['t_soc22']),  # L2-R2
            (0, 12, p['t12'], p['t_soc12']),  # L1-R2
            (4, 8, p['t12'], p['t_soc12'])    # L2-R1
        ]
        
        for src, dst, t, t_soc in interdot_pairs:
            add_interdot_tunneling(h, (src, dst), t, t_soc)

        return h
    
    def _build_interaction_dict(self):
        """
        Constructs the interaction Hamiltonian dictionary for the double quantum dot system
        in the extended 16-state basis, accounting for all orbital combinations (L1/L2/R1/R2).

        
        """
        p = self.parameters
        U0, U1, X, A, P = p['U0'], p['U1'], p['X'], p['A'], p['P']
        J, g_ortho, g_zz, g_z0, g_0z = p['J'], p['g_ortho'], p['g_zz'], p['g_z0'], p['g_0z']

        # Base orbital offsets: L1=0, L2=4, R1=8, R2=12
        orbital_offsets = {'L1': 0, 'L2': 4, 'R1': 8, 'R2': 12}
        V = {}

        # Helper function to generate intra-orbital terms for a given offset
        def add_intra_orbital_terms(offset):
            V.update({
                (0+offset, 1+offset, 1+offset, 0+offset): U0 + J*(g_zz + g_z0 + g_0z),
                (2+offset, 3+offset, 3+offset, 2+offset): U0 + J*(g_zz + g_z0 + g_0z),
                (0+offset, 2+offset, 2+offset, 0+offset): U0 + J*(g_zz - g_z0 - g_0z),
                (0+offset, 3+offset, 3+offset, 0+offset): U0 + J*(g_zz - g_z0 - g_0z),
                (1+offset, 2+offset, 2+offset, 1+offset): U0 + J*(g_zz - g_z0 - g_0z),
                (1+offset, 3+offset, 3+offset, 1+offset): U0 + J*(g_zz - g_z0 - g_0z),
                (0+offset, 2+offset, 0+offset, 2+offset): 4*J*g_ortho,
                (0+offset, 3+offset, 1+offset, 2+offset): 4*J*g_ortho,
                (1+offset, 3+offset, 1+offset, 3+offset): 4*J*g_ortho,
                (1+offset, 2+offset, 0+offset, 3+offset): 4*J*g_ortho
            })

        # Add intra-orbital terms for all orbitals
        for offset in orbital_offsets.values():
            add_intra_orbital_terms(offset)

        # Helper function to generate inter-orbital terms between two orbitals
        def add_inter_orbital_terms(src_offset, dst_offset):
            # Direct interaction terms (U1)
            for i in range(4):  # All spin/valley combinations
                for j in range(4):
                    V[(i+src_offset, j+dst_offset, j+dst_offset, i+src_offset)] = U1

            # Pair hopping (P) and exchange (X) terms
            pairs = [
                (0,1,1,0), (0,2,2,0), (0,3,3,0), 
                (1,2,2,1), (1,3,3,1), (2,3,3,2)
            ]
            for (r1, r2, l1, l2) in pairs:
                V[(r1+src_offset, r2+src_offset, l1+dst_offset, l2+dst_offset)] = P
                V[(r1+src_offset, l1+dst_offset, r1+src_offset, l1+dst_offset)] = X

            # Density-assisted hopping (A)
            a_pairs_type2 = [
                (1,0,0,1), (2,0,0,2), (2,1,1,2), (3,0,0,3), (3,2,2,3), (3,1,1,3)
            ]
            for (r, l, rp, lp) in pairs:
                V[(r+src_offset, l+src_offset, rp+dst_offset, lp+src_offset)] = A
                V[(r+src_offset, l+dst_offset, rp+src_offset, lp+dst_offset)] = A

            for (r, l, rp, lp) in a_pairs_type2:
                V[(r+src_offset, l+dst_offset, rp+src_offset, lp+dst_offset)] = A

        # Add inter-orbital terms between all L-R combinations
        for src in ['L1', 'L2']:
            for dst in ['R1', 'R2']:
                add_inter_orbital_terms(orbital_offsets[src], orbital_offsets[dst])

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
        orb1 = '1' if '1' in lab1 else '2'
        orb2 = '1' if '1' in lab2 else '2'

        orb_part1 = lab1.replace('Up', '').replace('Down', '').replace('+', '').replace('-', '').replace('1', '').replace('2', '')
        orb_part2 = lab2.replace('Up', '').replace('Down', '').replace('+', '').replace('-', '').replace('1', '').replace('2', '')

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

        # --- ENERGY ORBITAL ---
        orb_type = orb1 + orb2

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
            "Orbital Type": orb_type,
            "Orbital Symmetry": symmetry,
        }
    
    def create_determinant_from_labels(self, labels: list) -> int:
        """
        Creates a determinant from a list of orbital labels.
        The labels should be in the format 'L1Up+', 'L1Down+', etc.
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

    

def represent_B_field_dependence(arrayToPlot, fixedParameters=None, number_of_eigenstates=-1):
    dqd = DQD_spin_valley_singleOrbital_twoParticles()

    if number_of_eigenstates == -1:
        number_of_eigenstates = len(dqd.basis)

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

    plt.figure()
    for i in range(number_of_eigenstates):
        plt.plot(bFields, eigvals[:, i], label=f'Eigenstate {i+1}')
    plt.xlabel('B-field (T)')
    plt.ylabel('Eigenvalue (meV)')
    plt.title('Eigenvalues of DQD Spin-Valley System')
    plt.grid(True)
    plt.show()


def represent_detuning_dependence(arrayToPlot, fixedParameters=None, number_of_eigenstates=-1):
    dqd = DQD_spin_valley_singleOrbital_twoParticles()
    if number_of_eigenstates == -1:
        number_of_eigenstates = len(dqd.basis)

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

def represent_DeltaSO_dependence(arrayToPlot, fixedParameters=None, number_of_eigenstates=-1):
    dqd = DQD_spin_valley_singleOrbital_twoParticles()
    if number_of_eigenstates == -1:
        number_of_eigenstates = len(dqd.basis)

    DeltaSO_values = arrayToPlot if arrayToPlot is not None else np.linspace(0.0, 0.1, 100)  # Default range for DeltaSO
    eigvals = np.zeros((len(DeltaSO_values), len(dqd.basis)), dtype=np.complex128)
    eigvectors = np.zeros((len(DeltaSO_values), len(dqd.basis), len(dqd.basis)), dtype=np.complex128)

    for i, DeltaSO in enumerate(DeltaSO_values):
        parameters_to_change = fixedParameters.copy() if fixedParameters is not None else {}
        parameters_to_change[DQDParameters.DELTA_SO.value] = DeltaSO
        eigval, eigv = dqd.calculate_eigenvalues_and_eigenvectors(parameters_to_change=parameters_to_change)
        eigvals[i] = eigval
        eigvectors[i] = eigv

    plt.figure()
    for i in range(number_of_eigenstates):
        plt.plot(DeltaSO_values, eigvals[:, i], label=f'Eigenstate {i+1}')
    plt.xlabel('Delta SO (meV)')
    plt.ylabel('Eigenvalue (meV)')
    plt.title('Eigenvalues of DQD Spin-Valley System')
    plt.grid(True)
    plt.show()

def zero_field_classification(fixedParameters=None, number_of_eigenstates=-1):
    dqd = DQD_spin_valley_singleOrbital_twoParticles()
    if number_of_eigenstates == -1:
        number_of_eigenstates = len(dqd.basis)
    parameters_to_change = fixedParameters.copy() if fixedParameters is not None else {}
    parameters_to_change[DQDParameters.B_FIELD.value] = 0.1  # Set a small B-field to avoid degeneracy
    eigval, eigv = dqd.calculate_eigenvalues_and_eigenvectors(fixedParameters)
    
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
        DQDParameters.E_I.value: 100.0,  # Set detuning to
        DQDParameters.T11.value: 0.04,  # Set hopping parameter
        DQDParameters.T12.value: 0.04,  # Set hopping parameter
        DQDParameters.T22.value: 0.04,  # Set hopping parameter
        DQDParameters.DELTA_SO.value: -0.04,  # Set Kane
        DQDParameters.DELTA_KK.value: 0.02,  # Set valley mixing
        DQDParameters.DELTA_ORB.value: 10.0,  # Set orbital splitting
        DQDParameters.T_SOC11.value: 0.0,  # Set spin-flip
        DQDParameters.T_SOC12.value: 0.0,  # Set spin-flip
        DQDParameters.T_SOC22.value: 0.0,  # Set spin-flip
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

    number_of_eigenstates = 30  # Number of eigenstates to plot
    arrayToPlot = np.linspace(0.0, 5.0, 100)  # Example range for B-field or detuning
    zero_field_classification(fixedParameters=fixedParameters, number_of_eigenstates=number_of_eigenstates)
    represent_B_field_dependence(arrayToPlot=arrayToPlot, fixedParameters=fixedParameters, number_of_eigenstates=number_of_eigenstates)