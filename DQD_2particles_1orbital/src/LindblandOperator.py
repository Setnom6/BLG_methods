import numpy as np
from src.FockSpaceUtilities import FockSpaceUtilities


import numpy as np
from typing import Optional
from scipy.sparse import dok_matrix, csc_matrix


class LindbladOperator:
    
    def __init__(self, fockUtils: FockSpaceUtilities):
        self.fockUtils = fockUtils
        self.basis = fockUtils.basis
        self.norb = fockUtils.norb
        self.indexMap = {state: idx for idx, state in enumerate(self.basis)}

    def buildDephasingOperator(self, orbital: int) -> csc_matrix:
        """
        Build the dephasing Lindblad operator: L = n_i = c_i† c_i

        Args:
            orbital: index of the orbital where dephasing occurs

        Returns:
            csc_matrix representing the Lindblad operator in Fock basis
        """
        size = len(self.basis)
        op = dok_matrix((size, size), dtype=complex)

        for i, state in enumerate(self.basis):
            if (state >> orbital) & 1:
                op[i, i] = 1.0

        return op.tocsc()

    def buildDecoherenceOperator(self, fromOrbital: int, toOrbital: int) -> csc_matrix:
        """
        Build the decoherence Lindblad operator: L = c_to† c_from

        Args:
            fromOrbital: orbital to annihilate (j)
            toOrbital: orbital to create (i)

        Returns:
            csc_matrix representing the Lindblad operator in Fock basis
        """
        size = len(self.basis)
        op = dok_matrix((size, size), dtype=complex)

        for i, state in enumerate(self.basis):
            newState, phase1 = self.fockUtils.apply_annihilation(state, fromOrbital)
            if newState is None:
                continue

            finalState, phase2 = self.fockUtils.apply_creation(newState, toOrbital)
            if finalState is None:
                continue

            j = self.indexMap.get(finalState)
            if j is not None:
                op[j, i] = phase1 * phase2

        return op.tocsc()

