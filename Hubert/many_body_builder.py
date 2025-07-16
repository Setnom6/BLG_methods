"""many_body_builder.py
-----------------------
Utility for building dense many‑body Hamiltonians from **user‑supplied single‑
particle and two‑body dictionaries** in *condensed‑matter index ordering*.

### Two‑body index convention (condensed‑matter)
We now follow the notation most common in condensed‑matter physics

    V[p, r, q, s]  ≡  ⟨ p  r | V | q  s ⟩ ,

i.e. the first **bra** electron indices are *(p, r)*, the **ket** indices are
*(q, s)*.  With this convention

* **On‑site Hubbard U** for opposite spins in the same orbital is
  `(0, 1, 1, 0)`.
* **Exchange / spin‑flip term** is `(0, 1, 0, 1)`. 
This is a virtual process in which state 0 indicates first orbital spin down and state 1 indicates first orbital spin up.
Then, the global state does not change (is antisymmetric). it just meausure the exchange J interactrion. 
In the on-site Hubbard U case the repulson due to Coulomb interaction is measured.

The library automatically Hermitian‑symmetrises:

``V[p, r, q, s] = conj( V[q, s, p, r] )``.

Users still only need to supply *one* representative per Hermitian pair.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from scipy.linalg import eigh
import numpy as np


OneBodyDict = Dict[Tuple[int, int], complex]
TwoBodyDict = Dict[Tuple[int, int, int, int], complex]
BitDet = int

__all__ = [
    "complete_hermitian_1body",
    "coulomb_tensor_from_dict",
    "generate_fock_basis",
    "ManyBodyHamiltonian",
]

#  One‑body helpers

def complete_hermitian_1body(hdict: OneBodyDict, norb: int) -> np.ndarray:
    """
    Complete a Hermitian one-body operator matrix given its upper triangular part.

    This function takes a dictionary representing the upper triangular part
    of a Hermitian matrix in one-body operator notation and reconstructs the
    complete Hermitian matrix as a NumPy array. In Hermitian matrices, the
    element at position `(p, q)` must equal the complex conjugate of the element
    at position `(q, p)`. The function ensures that this property is preserved
    while completing the full matrix from the given input data.

    The input dictionary `hdict` should only contain keys where `p ≤ q`, as it
    represents the upper triangular part. If any key is found with `p > q`,
    a `ValueError` is raised. This ensures the validity of the input format.

    :param hdict: A dictionary where keys are tuples `(p, q)` with `p ≤ q`
                  and values are the corresponding matrix elements. Represents
                  the upper triangular part of a Hermitian matrix.
    :type hdict: OneBodyDict
    :param norb: The total number of orbitals, defining the dimensions of the
                 resulting matrix as `norb x norb`.
    :type norb: int
    :return: A complex-valued Hermitian matrix of shape `(norb, norb)` constructed
             from the input dictionary.
    :rtype: numpy.ndarray
    :raises ValueError: If a key `(p, q)` in the input dictionary satisfies `p > q`.
    """
    H = np.zeros((norb, norb), dtype=np.complex128)
    for (p, q), val in hdict.items():
        if p > q:
            raise ValueError("One‑body dict must only contain keys with p ≤ q")
        H[p, q] = val
        if p != q:
            H[q, p] = np.conjugate(val)
    return H

###############################################################################
#  Two‑body helpers (condensed‑matter ordering) ###############################
###############################################################################

def coulomb_tensor_from_dict(cdict: TwoBodyDict, norb: int) -> np.ndarray:
    """
    Generates the Coulomb tensor in the two-electron integral format from a given
    two-body dictionary representation. The resulting tensor includes both the
    provided integrals and their Hermitian counterparts by ensuring Hermitian
    symmetry.

    :param cdict: Dictionary representation of two-body integrals. The keys are
        tuples of orbital indices (p, r, q, s), and the values are the corresponding
        complex integral values.
    :type cdict: TwoBodyDict
    :param norb: Total number of orbitals. Determines the size of the resulting
        Coulomb tensor.
    :type norb: int
    :return: A 4-dimensional NumPy array of shape (norb, norb, norb, norb),
        representing the Coulomb tensor. The tensor is Hermitian with respect to
        its bra and ket indices.
    :rtype: numpy.ndarray
    """
    V = np.zeros((norb, norb, norb, norb), dtype=np.complex128)
    for (p, r, q, s), val in cdict.items():
        # Hermitian counterpart (bra ↔ ket)
        V[p, r, q, s] = val
        V[q, s, p, r] = np.conjugate(val)
    return V

###############################################################################
#  Fock‑space utilities ########################################################
###############################################################################

def popcount(x: int) -> int:
    """
    Counts the number of 1-bits in the binary representation of an integer.

    This function takes an integer as input and returns the count of
    bits that are set to 1 in its binary representation.

    :param x: An integer whose binary 1-bits are to be counted.
    :type x: int
    :return: The count of 1-bits in the binary representation of the input integer.
    :rtype: int
    """
    return x.bit_count()

def bits_before(state: int, orb: int) -> int:
    """
    Calculate the number of bits set to 1 in the binary representation of the input
    `state` up to (but not including) the given bit position `orb`.

    :param state: Integer representing the binary state from which the number of
        set bits will be counted.
    :type state: int
    :param orb: The bit position up to which the bits are considered (exclusive).
    :type orb: int
    :return: The count of bits set to 1 up to the specified position `orb`.
    :rtype: int
    """
    mask = (1 << orb) - 1
    return popcount(state & mask)

def apply_annihilation(state: BitDet, orb: int) -> Tuple[Optional[BitDet], int]:
    """
    Applies the annihilation operator to a given quantum state. The annihilation
    operator acts on a specific orbital in the state. If the orbital is not
    occupied, the function returns None and a phase of 0. Otherwise, it modifies
    the state by annihilating the particle in the specified orbital and computes
    the corresponding phase factor.

    :param state: The current quantum state represented as a bit determinant.
    :type state: BitDet

    :param orb: The orbital index on which the annihilation operator is applied.
    :type orb: int

    :return: A tuple where the first element is the modified quantum state
        represented as a bit determinant, or None if the orbital is not occupied,
        and the second element is the phase factor.
    :rtype: Tuple[Optional[BitDet], int]
    """
    if not (state >> orb & 1):
        return None, 0
    phase = (-1) ** bits_before(state, orb)
    return state & ~(1 << orb), phase

def apply_creation(state: BitDet, orb: int) -> Tuple[Optional[BitDet], int]:
    """
    Apply a creation operator on the given quantum bit state.

    This function takes a quantum bit state (state) and applies a creation
    operator to the specified orbital (orb). If the orbital is already
    occupied (has a bit value of 1), the function will return None and a
    phase of 0. If the orbital is unoccupied (bit value of 0), the function
    calculates the resulting state after applying the creation operator and
    the associated phase factor.

    :param state: A quantum bit state represented as a `BitDet` type. Each bit
        represents the occupation state of an orbital.
    :param orb: The orbital index where the creation operator is applied.
        It must be a non-negative integer corresponding to the position of the
        bit in the quantum bit state.
    :return: A tuple containing the resulting quantum bit state after applying
        the creation operator and the associated phase factor. If the operation is
        invalid due to the orbital already being occupied, the function returns
        (None, 0).
    """
    if state >> orb & 1:
        return None, 0
    phase = (-1) ** bits_before(state, orb)
    return state | (1 << orb), phase

def generate_fock_basis(norb: int, n_elec: Optional[int | Sequence[int]] = None) -> List[BitDet]:
    """
    Generates the Fock basis based on the number of orbitals and a specified number
    of electrons. The function computes all possible bit determinants that meet
    the criteria for the given input parameters.

    :param norb: Number of orbitals. Must be a non-negative integer.
    :type norb: int
    :param n_elec: Number of electrons or a sequence of allowed numbers of
        electrons. If None, all possible configurations are considered.
        Can be an integer, a sequence of integers, or None.
    :type n_elec: Optional[int | Sequence[int]]
    :return: A list of bit determinants representing the Fock basis. Each bit
        determinant is an integer whose binary representation encodes the
        occupation of orbitals.
    :rtype: List[BitDet]
    """
    allowed = (
        set(range(norb + 1))
        if n_elec is None
        else {n_elec} if isinstance(n_elec, int) else set(n_elec)
    )
    return [det for det in range(1 << norb) if popcount(det) in allowed]

def get_occupied_orbitals(state: int, norb: int) -> list[int]:
            return [i for i in range(norb) if (state >> i) & 1]

#  Many‑body Hamiltonian

class ManyBodyHamiltonian:
    """
    Represents a many-body Hamiltonian operator.

    This class is designed to construct and manipulate the Hamiltonian of a quantum
    many-body system, using one-body and two-body interaction terms. The Hamiltonian
    matrix can be generated based on a specific basis of determinant. It provides tools to construct the
    system's basis and build the resulting sparse Hamiltonian matrix.

    :ivar norb: Number of orbitals in the system.
    :type norb: int
    :ivar h1: Hermitian one-body Hamiltonian matrix.
    :type h1: np.ndarray
    :ivar V: Two-body Coulomb interaction tensor.
    :type V: np.ndarray
    :ivar matrix: The resulting built Hamiltonian matrix; `None` if not yet built.
    :type matrix: Optional[np.ndarray]
    """

    def __init__(self, norb: int, hdict: OneBodyDict, cdict: TwoBodyDict):
        self.norb = norb
        self.h1 = complete_hermitian_1body(hdict, norb)
        self.V = coulomb_tensor_from_dict(cdict, norb)
        self.matrix: Optional[np.ndarray] = None

    def generate_basis(self, n_elec: Optional[int | Sequence[int]] = None):
        return generate_fock_basis(self.norb, n_elec)

    def build(self, basis: Sequence[BitDet]):
        dim = len(basis)
        H = np.zeros((dim, dim), dtype=np.complex128)
        idx = {det: i for i, det in enumerate(basis)}

        # one‑body part
        for p in range(self.norb):
            for q in range(self.norb):
                hval = self.h1[p, q]
                if hval == 0:
                    continue
                for j, det in enumerate(basis):
                    t1, s1 = apply_annihilation(det, q)
                    if t1 is None:
                        continue
                    t2, s2 = apply_creation(t1, p)
                    if t2 is None:
                        continue
                    H[idx[t2], j] += hval * s1 * s2

        # two‑body part
        for p, r, q, s in itertools.product(range(self.norb), repeat=4):
            v = self.V[p, r, q, s]
            if v == 0:
                continue
            for j, det in enumerate(basis):
                t1, s1 = apply_annihilation(det, s)
                if t1 is None:
                    continue
                t2, s2 = apply_annihilation(t1, q)
                if t2 is None:
                    continue
                t3, s3 = apply_creation(t2, r)
                if t3 is None:
                    continue
                t4, s4 = apply_creation(t3, p)
                if t4 is None:
                    continue
                H[idx[t4], j] += 0.5 * v * s1 * s2 * s3 * s4

        self.matrix = (H + H.T.conj()) / 2.0
        return self.matrix

def single_triplet_example():
    """
    Generates and visualizes a many-body Hamiltonian with two spin-orbitals in one spatial orbital,
    showing its eigenvalues as a function of a varying parameter and printing basis determinants
    and the dense Hamiltonian matrix.

    The code constructs a many-body Hamiltonian using the given one-body terms and Coulomb interaction
    parameters. It generates a basis of determinants for two particles, solves the eigenvalue problem
    provided by the Hamiltonian within this basis for a range of parameters, and visualizes the results.

    Attributes:
        h (dict): Dictionary representing the one-body null terms of the Hamiltonian.
        V (dict): Dictionary representing the Coulomb interaction terms of the Hamiltonian.
        H (ManyBodyHamiltonian): Instance of the ManyBodyHamiltonian defining the system configuration.
        basis (list): Basis determinants generated for the given particle number.
        eps (numpy.ndarray): Parameter range for varying a one-body parameter in the Hamiltonian.
        eigenvals (numpy.ndarray): Array containing eigenvalues of the Hamiltonian for each parameter value.

    :param:
        None

    :raises:
        None

    :return:
        None
    """
    # Two spin‑orbitals ↑, ↓ in one spatial orbital

    # One‑body null
    h = {(0, 0): 0.1,
         (1, 1): -0.1,
         (2, 2): 1.0 + 0.1,
         (3, 3): 1.0 - 0.1,
         (0, 2): 0.02,
         (1, 3): 0.02}

    # On‑site Hubbard U in condensed‑matter ordering: (0,1,1,0)
    V = {(0, 1, 1, 0): 2.0,
         (0, 2, 2, 0): 1.0,
         (0, 3, 3, 0): 1.0,
         (1, 2, 2, 1): 1.0,
         (1, 3, 3, 1): 1.0,
         (2, 3, 3, 2): 2.0,
         (0, 2, 0, 2): -0.0,
         (1, 3, 1, 3): -0.0,
         }
    
    H = ManyBodyHamiltonian(4, h, V)
    basis = H.generate_basis(2)
    eps = np.linspace(0.0, 2.5, 1000)
    eigenvals = np.zeros((len(eps), len(basis)), dtype=np.complex128)
    for i, epsi in enumerate(eps):
        h[(2, 2)] = epsi + 0.1
        h[(3, 3)] = epsi - 0.1
        H = ManyBodyHamiltonian(4, h, V)
        basis = H.generate_basis(2)
        H.build(basis)
        eigval, eigv = eigh(H.matrix)
        eigenvals[i] = eigval
    plt.figure()
    for i in range(len(basis)):
        plt.scatter(eps, eigenvals[:, i], s=0.5)
    #plt.xlim(0.5, 1.5)
    plt.ylim(0, 3)
    plt.show()

    print("Basis determinants:", [bin(b) for b in basis])
    print("Dense matrix :\n", H.matrix.real)

def double_orbital_spin_valley_double_dot():
    """
    This function demonstrates the construction of a Hamiltonian model for a double electron
    spin and valley-coupled quantum dot system, including various energy contributions arising
    from external magnetic fields, spin-orbit couplings, interaction terms, and kinetic contributions.
    The resulting model can be used for quantum simulations and analysis in quantum systems research.

    The Hamiltonian is constructed using the following key components:
        - Local energy splittings: orbital, spin Zeeman, valley Zeeman, and Kane-Mele splittings.
        - Non-local hopping terms between basis states.
        - Non-local interaction terms including spin and valley exchange interactions, local
          and non-local Coulomb interactions.
        - External magnetic field effects on spin and valley Zeeman splittings.

    The functionality includes:
        - Initialization of single-particle and interaction terms.
        - Definition of spin-valley-orbital coupling.
        - Construction of the many-body Hamiltonian via matrix representations of the defined terms.
        - Basis generation for simulating the system.

    This function provides a customizable starting point for detailed studies of many-body physics
    in quantum systems under combined spin, valley, and orbital degrees of freedom.

    :param None: There are no input arguments for this function.

    :return: None
    """
    # One‑body null
    b_field = 0.05  # in Tesla
    in_plane = 0.0
    mub = 0.05788  # meV/T
    gs = 2.0
    gv = 20

    u = 6
    nearest_neighbour = u / 4
    exchange = u / 100
    g_ortho = 90.0
    g_zz = 4.0
    g_zo = -90.0
    g_oz = -90.0
    cursive_I = 4.0 * 10 ** -4

    u_sr_v_ex = u + cursive_I * (g_zz + g_oz + g_zo)
    u_sr_s_ex = u + cursive_I * (g_zz - (g_oz + g_zo))
    vx = 4 * cursive_I * g_ortho

    eps_i = 0
    orbital_splitting = 4
    spin_zeeman_splitting = 0.5 * gs * mub * b_field
    kane_mele_splitting = 0.06
    valley_zeeman_splitting = 0.5 * gv * mub * b_field

    t = 0.1
    t_soc = 0.00
    delta_kk = 0.00

    h = {(0, 0): spin_zeeman_splitting + valley_zeeman_splitting - 0.5 * kane_mele_splitting,  # R1 up K
         (1, 1): valley_zeeman_splitting - spin_zeeman_splitting + 0.5 * kane_mele_splitting,  # R1 down K
         (2, 2): spin_zeeman_splitting - valley_zeeman_splitting + 0.5 * kane_mele_splitting,  # R1 up K'
         (3, 3): -1 * valley_zeeman_splitting - spin_zeeman_splitting - 0.5 * kane_mele_splitting,  # R1 down K'
         (4, 4): orbital_splitting + spin_zeeman_splitting + valley_zeeman_splitting - 0.5 * kane_mele_splitting,# R2 up K
         (5, 5): orbital_splitting + valley_zeeman_splitting - spin_zeeman_splitting + 0.5 * kane_mele_splitting,# R2 down K
         (6, 6): orbital_splitting + spin_zeeman_splitting - valley_zeeman_splitting + 0.5 * kane_mele_splitting,# R2 up K'
         (7, 7): orbital_splitting + -1 * valley_zeeman_splitting - spin_zeeman_splitting - 0.5 * kane_mele_splitting,# R2 down K'
         (8, 8): eps_i + spin_zeeman_splitting + valley_zeeman_splitting - 0.5 * kane_mele_splitting,  # L1 up K
         (9, 9): eps_i + valley_zeeman_splitting - spin_zeeman_splitting + 0.5 * kane_mele_splitting,  # L1 down K
         (10, 10): eps_i + spin_zeeman_splitting - valley_zeeman_splitting + 0.5 * kane_mele_splitting,  # L1 up K'
         (11, 11): eps_i + -1 * valley_zeeman_splitting - spin_zeeman_splitting - 0.5 * kane_mele_splitting,# L1 down K'
         (12, 12): eps_i + orbital_splitting + spin_zeeman_splitting + valley_zeeman_splitting - 0.5 * kane_mele_splitting,# L2 up K
         (13, 13): eps_i + orbital_splitting + valley_zeeman_splitting - spin_zeeman_splitting + 0.5 * kane_mele_splitting,# L2 down K
         (14, 14): eps_i + orbital_splitting + spin_zeeman_splitting - valley_zeeman_splitting + 0.5 * kane_mele_splitting,# L2 up K'
         (15, 15): eps_i + orbital_splitting + -1 * valley_zeeman_splitting - spin_zeeman_splitting - 0.5 * kane_mele_splitting,# L2 down K'
         }

    for i in range(16):
        for j in range(16):
            if j == i + 8:
                h[(i, j)] = t
    V = {
        (0, 1, 1, 0): u_sr_s_ex,
        (0, 2, 2, 0): u_sr_v_ex,
        (0, 3, 3, 0): u_sr_v_ex,
        (0, 4, 4, 0): u,
        (0, 5, 5, 0): u,
        (0, 6, 6, 0): u,
        (0, 7, 7, 0): u,
        (1, 2, 2, 1): u_sr_v_ex,
        (1, 3, 3, 1): u_sr_v_ex,
        (1, 4, 4, 1): u,
        (1, 5, 5, 1): u,
        (1, 6, 6, 1): u,
        (1, 7, 7, 1): u,
        (2, 3, 3, 2): u_sr_s_ex,
        (2, 4, 4, 2): u,
        (2, 5, 5, 2): u,
        (2, 6, 6, 2): u,
        (2, 7, 7, 2): u,
        (3, 4, 4, 3): u,
        (3, 5, 5, 3): u,
        (3, 6, 6, 3): u,
        (3, 7, 7, 3): u,
        (4, 5, 5, 4): u_sr_s_ex,
        (4, 6, 6, 4): u_sr_v_ex,
        (4, 7, 7, 4): u_sr_v_ex,
        (5, 6, 6, 5): u_sr_v_ex,
        (5, 7, 7, 5): u_sr_v_ex,
        (6, 7, 7, 6): u_sr_s_ex,
        (0, 2, 0, 2): vx,
        (0, 3, 1, 2): vx,
        (1, 3, 1, 3): vx,
        (1, 2, 0, 3): vx,
        (4, 6, 4, 6): vx,
        (4, 7, 5, 6): vx,
        (5, 7, 5, 7): vx,
        (5, 6, 4, 7): vx
    }
    exchange_pairs = [(0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15)]
    for r in range(8):
        for l in range(8, 16):
            if r <= l:
                V[(r, l, l, r)] = nearest_neighbour

    for (r, l) in exchange_pairs:
        if (r, l) < (l, r):
            V[(r, l, r, l)] = exchange / 2

    for j in range(16):  # assisting orbital (density)
        for i in range(16):  # hopping target
            for k in range(16):  # hopping source
                if i != k and (i, j) <= (k, j):  # upper triangle only
                    V[(i, j, k, j)] = exchange

    eps = np.arange(-2, 5, 0.01)
    bfields = np.arange(0, 1, 0.01)

    H = ManyBodyHamiltonian(16, h, V)
    basis = H.generate_basis(2)
    eigenvals = np.zeros((len(bfields), len(basis)), dtype=np.complex128)

    for i, b_field in enumerate(bfields):
        print("B-field:", b_field)
        spin_zeeman_splitting = 0.5 * gs * mub * b_field
        kane_mele_splitting = 0.06
        valley_zeeman_splitting = 0.5 * gv * mub * b_field

        h[(0, 0)] = spin_zeeman_splitting + valley_zeeman_splitting - 0.5 * kane_mele_splitting
        # R1 down K
        h[(1, 1)] = valley_zeeman_splitting - spin_zeeman_splitting + 0.5 * kane_mele_splitting
        # R1 up K'
        h[(2, 2)] = spin_zeeman_splitting - valley_zeeman_splitting + 0.5 * kane_mele_splitting
        # R1 down K'
        h[(3, 3)] = -1 * valley_zeeman_splitting - spin_zeeman_splitting - 0.5 * kane_mele_splitting

        # R2 up K
        h[(4, 4)] = orbital_splitting + spin_zeeman_splitting + valley_zeeman_splitting - 0.5 * kane_mele_splitting
        # R2 down K
        h[(5, 5)] = orbital_splitting + valley_zeeman_splitting - spin_zeeman_splitting + 0.5 * kane_mele_splitting
        # R2 up K'
        h[(6, 6)] = orbital_splitting + spin_zeeman_splitting - valley_zeeman_splitting + 0.5 * kane_mele_splitting
        # R2 down K'
        h[(7, 7)] = orbital_splitting + -1 * valley_zeeman_splitting - spin_zeeman_splitting - 0.5 * kane_mele_splitting

        # L1 up K
        h[(8, 8)] = eps_i + spin_zeeman_splitting + valley_zeeman_splitting - 0.5 * kane_mele_splitting
        # L1 down K
        h[(9, 9)] = eps_i + valley_zeeman_splitting - spin_zeeman_splitting + 0.5 * kane_mele_splitting
        # L1 up K'
        h[(10, 10)] = eps_i + spin_zeeman_splitting - valley_zeeman_splitting + 0.5 * kane_mele_splitting
        # L1 down K'
        h[(11, 11)] = eps_i + -1 * valley_zeeman_splitting - spin_zeeman_splitting - 0.5 * kane_mele_splitting

        # L2 up K
        h[(12,
           12)] = eps_i + orbital_splitting + spin_zeeman_splitting + valley_zeeman_splitting - 0.5 * kane_mele_splitting
        # L2 down K
        h[(13,
           13)] = eps_i + orbital_splitting + valley_zeeman_splitting - spin_zeeman_splitting + 0.5 * kane_mele_splitting
        # L2 up K'
        h[(14,
           14)] = eps_i + orbital_splitting + spin_zeeman_splitting - valley_zeeman_splitting + 0.5 * kane_mele_splitting
        # L2 down K'
        h[(
            15,
            15)] = eps_i + orbital_splitting + -1 * valley_zeeman_splitting - spin_zeeman_splitting - 0.5 * kane_mele_splitting

        H = ManyBodyHamiltonian(16, h, V)
        basis = H.generate_basis(2)
        H.build(basis)
        eigval, eigv = eigh(H.matrix)
        eigenvals[i] = eigval
    plt.figure()
    for i in range(len(basis)):
        plt.scatter(bfields, eigenvals[:, i], s=0.5)
    # plt.xlim(1.25, 1.75)
    plt.ylim(-4, 5)
    plt.show()

    print("Basis determinants:", [bin(b) for b in basis])
    print("Dense matrix :\n", H.matrix.real)

def spin_valley_double_dot():
    """
    Simulates the spin-valley coupling in a double quantum dot system with specific physical effects,
    such as spin-orbit interaction, valley Zeeman effect, and Kane-Mele splitting. It comprises the
    construction of the single-particle Hamiltonian, on-site Coulomb potentials, inter-site interactions,
    and the many-body Hamiltonian. The eigenvalues are calculated for varying input parameters.

    Summary:
    This function defines and computes the Hamiltonian of a spin-valley double quantum dot, incorporating
    effects such as spin Zeeman splitting, valley Zeeman splitting, orbital splitting, and exchange interaction.
    A basis is generated for representing the many-body states, and the Hamiltonian eigenvalues and eigenvectors
    are solved using numerical diagonalization.

    :param b_field: Magnetic field in Tesla
    :type b_field: float

    :param in_plane: Magnetic field component in-plane, initially unused
    :type in_plane: float

    :param mub: Bohr magneton value in meV/T
    :type mub: float

    :param gs: Spin g-factor
    :type gs: float

    :param gv: Valley g-factor
    :type gv: float

    :param u: Onsite Coulomb repulsion
    :type u: float

    :param nearest_neighbour: Coulomb interaction between nearest neighbor sites
    :type nearest_neighbour: float

    :param exchange: Exchange interaction between quantum dots
    :type exchange: float

    :param g_ortho: Orthogonal component of tunneling corrections
    :type g_ortho: float

    :param g_zz: Correction along Z-axis
    :type g_zz: float

    :param g_zo: Correction for Z-O mixing
    :type g_zo: float

    :param g_oz: Correction for O-Z mixing
    :type g_oz: float

    :param cursive_I: Renormalization parameter for Coulomb corrections
    :type cursive_I: float

    :param orbital_splitting: Difference due to orbital splitting
    :type orbital_splitting: float

    :param spin_zeeman_splitting: Spin-derived energy splitting
    :type spin_zeeman_splitting: float

    :param kane_mele_splitting: Kane-Mele spin-orbit coupling splitting
    :type kane_mele_splitting: float

    :param valley_zeeman_splitting: Valley-derived energy splitting
    :type valley_zeeman_splitting: float

    :param t: Inter-dot tunneling amplitude
    :type t: float

    :param t_soc: Spin-orbit coupling tunneling term, set to zero
    :type t_soc: float

    :param delta_kk: Intervalley scattering correction, set to zero
    :type delta_kk: float

    :param h: Dictionary representing the single-particle Hamiltonian
    :type h: dict

    :param V: Dictionary representing the two-body interaction terms
    :type V: dict

    :param eps: Energy level steps for variation
    :type eps: numpy.ndarray

    :param bfields: Array of varying magnetic field strengths
    :type bfields: numpy.ndarray

    :param H: Many-body Hamiltonian object with 8 orbitals
    :type H: ManyBodyHamiltonian

    :param basis: Basis states defined for a 2-particle system
    :type basis: list

    :param eigenvals: Computed eigenvalues for varying energy levels
    :type eigenvals: numpy.ndarray

    :return: None
    """
    # One‑body null
    b_field = 0.5  # in Tesla
    in_plane = 0.0
    mub = 0.05788  # meV/T
    gs = 2.0
    gv = 20

    u = 6
    nearest_neighbour = u / 4
    exchange = u / 10
    g_ortho = 90.0
    g_zz = 4.0
    g_zo = -90.0
    g_oz = -90.0
    cursive_I = 4.0 * 10 ** -4

    u_sr_v_ex = u + cursive_I * (g_zz + g_oz + g_zo)
    u_sr_s_ex = u + cursive_I * (g_zz - (g_oz + g_zo))
    vx = 4 * cursive_I * g_ortho

    eps_i = 0
    orbital_splitting = 4
    spin_zeeman_splitting = 0.5 * gs * mub * b_field
    kane_mele_splitting = 0.06
    valley_zeeman_splitting = 0.5 * gv * mub * b_field

    t = 0.01
    t_soc = 0.00
    delta_kk = 0.00

    h = {(0, 0): spin_zeeman_splitting + valley_zeeman_splitting - 0.5 * kane_mele_splitting,  # R1 up K
         (1, 1): valley_zeeman_splitting - spin_zeeman_splitting + 0.5 * kane_mele_splitting,  # R1 down K
         (2, 2): spin_zeeman_splitting - valley_zeeman_splitting + 0.5 * kane_mele_splitting,  # R1 up K'
         (3, 3): -1 * valley_zeeman_splitting - spin_zeeman_splitting - 0.5 * kane_mele_splitting,  # R1 down K'
         (4, 4): orbital_splitting + spin_zeeman_splitting + valley_zeeman_splitting - 0.5 * kane_mele_splitting,
         # R2 up K
         (5, 5): orbital_splitting + valley_zeeman_splitting - spin_zeeman_splitting + 0.5 * kane_mele_splitting,
         # R2 down K
         (6, 6): orbital_splitting + spin_zeeman_splitting - valley_zeeman_splitting + 0.5 * kane_mele_splitting,
         # R2 up K'
         (7, 7): orbital_splitting + -1 * valley_zeeman_splitting - spin_zeeman_splitting - 0.5 * kane_mele_splitting,
         # R2 down K'
         }

    for i in range(8):
        for j in range(8):
            if j == i + 4:
                h[(i, j)] = t
    print(h)
    V = {
        (0, 1, 1, 0): u_sr_s_ex,
        (0, 2, 2, 0): u_sr_v_ex,
        (0, 3, 3, 0): u_sr_v_ex,
        (0, 4, 4, 0): nearest_neighbour,
        (0, 5, 5, 0): nearest_neighbour,
        (0, 6, 6, 0): nearest_neighbour,
        (0, 7, 7, 0): nearest_neighbour,
        (1, 2, 2, 1): u_sr_v_ex,
        (1, 3, 3, 1): u_sr_v_ex,
        (1, 4, 4, 1): nearest_neighbour,
        (1, 5, 5, 1): nearest_neighbour,
        (1, 6, 6, 1): nearest_neighbour,
        (1, 7, 7, 1): nearest_neighbour,
        (2, 3, 3, 2): u_sr_s_ex,
        (2, 4, 4, 2): nearest_neighbour,
        (2, 5, 5, 2): nearest_neighbour,
        (2, 6, 6, 2): nearest_neighbour,
        (2, 7, 7, 2): nearest_neighbour,
        (3, 4, 4, 3): nearest_neighbour,
        (3, 5, 5, 3): nearest_neighbour,
        (3, 6, 6, 3): nearest_neighbour,
        (3, 7, 7, 3): nearest_neighbour,
        (4, 5, 5, 4): u_sr_s_ex,
        (4, 6, 6, 4): u_sr_v_ex,
        (4, 7, 7, 4): u_sr_v_ex,
        (5, 6, 6, 5): u_sr_v_ex,
        (5, 7, 7, 5): u_sr_v_ex,
        (6, 7, 7, 6): u_sr_s_ex,
        (0, 2, 0, 2): vx,
        (0, 3, 1, 2): vx,
        (1, 3, 1, 3): vx,
        (1, 2, 0, 3): vx,
        (4, 6, 4, 6): vx,
        (4, 7, 5, 6): vx,
        (5, 7, 5, 7): vx,
        (5, 6, 4, 7): vx
    }
    exchange_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
    for (r, l) in exchange_pairs:
        if (r, l) < (l, r):
            V[(r, l, r, l)] = -1 * exchange / 2

    eps = np.arange(-5, 5, 0.01)
    bfields = np.arange(0, 1, 0.01)

    H = ManyBodyHamiltonian(8, h, V)
    basis = H.generate_basis(2)
    eigenvals = np.zeros((len(eps), len(basis)), dtype=np.complex128)

    for i, eps_i in enumerate(eps):
        spin_zeeman_splitting = 0.5 * gs * mub * b_field
        kane_mele_splitting = 0.06
        valley_zeeman_splitting = 0.5 * gv * mub * b_field

        h[(0, 0)] = spin_zeeman_splitting + valley_zeeman_splitting - 0.5 * kane_mele_splitting  # R1 up K
        h[(1, 1)] = valley_zeeman_splitting - spin_zeeman_splitting + 0.5 * kane_mele_splitting  # R1 down K
        h[(2, 2)] = spin_zeeman_splitting - valley_zeeman_splitting + 0.5 * kane_mele_splitting  # R1 up K'
        h[(3, 3)] = -1 * valley_zeeman_splitting - spin_zeeman_splitting - 0.5 * kane_mele_splitting  # R1 down K'
        h[(4, 4)] = eps_i + spin_zeeman_splitting + valley_zeeman_splitting - 0.5 * kane_mele_splitting  # R2 up K
        h[(5, 5)] = eps_i + valley_zeeman_splitting - spin_zeeman_splitting + 0.5 * kane_mele_splitting  # R2 down K
        h[(6, 6)] = eps_i + spin_zeeman_splitting - valley_zeeman_splitting + 0.5 * kane_mele_splitting  # R2 up K'
        h[(
        7, 7)] = eps_i + -1 * valley_zeeman_splitting - spin_zeeman_splitting - 0.5 * kane_mele_splitting  # R2 down K'

        H = ManyBodyHamiltonian(8, h, V)
        basis = H.generate_basis(2)
        H.build(basis)
        eigval, eigv = eigh(H.matrix)
        eigenvals[i] = eigval
    plt.figure()
    for i in range(len(basis)):
        plt.scatter(eps, eigenvals[:, i], s=0.5)
    # plt.xlim(3, 5)
    plt.ylim(4, 7)
    plt.show()

    print("Basis determinants:", [bin(b) for b in basis])
    print("Dense matrix :\n", H.matrix.real)


if __name__ == "__main__":
   single_triplet_example()

