from __future__ import annotations

import itertools
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from scipy.linalg import eigh
import numpy as np

OneBodyDict = Dict[Tuple[int, int], complex]
TwoBodyDict = Dict[Tuple[int, int, int, int], complex]

from .FockSpaceUtilities import FockSpaceUtilities

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
        self.h1 = self.complete_hermitian_1body(hdict, norb)
        self.V = self.coulomb_tensor_from_dict(cdict, norb)
        self.matrix: Optional[np.ndarray] = None
        self.FSUtils = None

    def generate_basis(self, n_elec: Optional[int | Sequence[int]] = None):
        self.FSUtils = FockSpaceUtilities(self.norb, n_elec)

    def build(self, n_elec=None):
        if self.FSUtils is None:
            self.generate_basis(n_elec)
        
        basis = self.FSUtils.basis
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
                    t1, s1 = self.FSUtils.apply_annihilation(det, q)
                    if t1 is None:
                        continue
                    t2, s2 = self.FSUtils.apply_creation(t1, p)
                    if t2 is None:
                        continue
                    H[idx[t2], j] += hval * s1 * s2

        # two‑body part
        for p, r, q, s in itertools.product(range(self.norb), repeat=4):
            v = self.V[p, r, q, s]
            if v == 0:
                continue
            for j, det in enumerate(basis):
                t1, s1 = self.FSUtils.apply_annihilation(det, s)
                if t1 is None:
                    continue
                t2, s2 = self.FSUtils.apply_annihilation(t1, q)
                if t2 is None:
                    continue
                t3, s3 = self.FSUtils.apply_creation(t2, r)
                if t3 is None:
                    continue
                t4, s4 = self.FSUtils.apply_creation(t3, p)
                if t4 is None:
                    continue
                H[idx[t4], j] += 0.5 * v * s1 * s2 * s3 * s4

        self.matrix = (H + H.T.conj()) / 2.0
        return self.matrix
    
    @staticmethod
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
    
    @staticmethod
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