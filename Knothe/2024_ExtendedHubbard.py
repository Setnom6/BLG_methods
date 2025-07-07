import numpy as np
from qutip import destroy, tensor, qeye, basis
from itertools import combinations
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

class BLGDQD_Hubbard:
    def __init__(self, params):
        """
        Extended Hubbard Hamiltonian for BLG DQD system according to Knothe 2024.
        It does not consider valley mixing DeltaKK' and assumes B is along the z-axis.
        It assumes only spin-conserving tunneling and no spin-orbit coupling between the dots (Extrinsic).
        
        Key parameters:
        - t: hopping between dots
        - U0: on-site interaction
        - U1: neighbor interaction
        - J: exchange integral
        - G_zz, G_z0, G_0z, G_perp: short-range couplings
        - X: inter-site exchange
        - A: density-assisted hopping
        - P: pair hopping
        - DeltaSO: spin-orbit splitting
        - gs: spin g-factor
        - gv: valley g-factor
        - muB: Bohr magneton
        """
        self.params = {
            't': 0.1,
            'U0': 10.0,
            'U1': 1.0,
            'J': 1.0, # meV/Gperp
            'G_zz': 0.75,
            'G_z0': 0.0495,
            'G_0z': 0.0495,
            'G_perp': 0.075,
            'X': 0.5,
            'A': 0.2,
            'P': 0.1,
            'DeltaSO': -0.04,  # meV
            'gs': 2.0,
            'gv': 35.0,
            'muB': 5.788e-2,  # meV/T
            'epsilon': [0.0, 0.0]  # dot energies
        }
        self.params.update(params)
        
        # Single-particle basis: 8 states indexed as in Appendix A
        # |0⟩ = |l:↑+⟩, |1⟩ = |l:↓+⟩, |2⟩ = |l:↑-⟩, |3⟩ = |l:↓-⟩
        # |4⟩ = |r:↑+⟩, |5⟩ = |r:↓+⟩, |6⟩ = |r:↑-⟩, |7⟩ = |r:↓-⟩
        self.dim_single = 8
        self.basis = self._construct_basis()
        self.stateToIndex = {state: i for i, state in enumerate(self.basis)}
        
        # Precompute interaction tensor
        self.uTensor = self._build_interaction_tensor()

        # Compute orbital splitting for checking
        DeltaOrb = - 0.5*(self.params['U0']-self.params['U1']) +0.5*np.sqrt((self.params['U0']-self.params['U1'])**2 + 16*self.params['t']**2)
        print(f"Orbital splitting (DeltaOrb): {DeltaOrb:.4f} meV")
    
    def _single_particle_hamiltonian(self, B):
        """Single-particle Hamiltonian including magnetic field effects."""
        p = self.params
        H = np.zeros((8, 8), dtype=np.complex128)
        
        # Diagonal terms (energies of each state)
        for dot in [0, 1]:  # 0=left, 1=right
            base = 4*dot
            # Base energy of the dot
            E0 = p['epsilon'][dot]
            
            # Zeeman terms and valley splitting
            H[base+0, base+0] = E0 + 0.5*p['DeltaSO'] + 0.5*p['muB']*B*(p['gv'] + p['gs'])  # l/r:↑+
            H[base+1, base+1] = E0 - 0.5*p['DeltaSO'] + 0.5*p['muB']*B*(p['gv'] - p['gs'])  # l/r:↓+
            H[base+2, base+2] = E0 - 0.5*p['DeltaSO'] + 0.5*p['muB']*B*(-p['gv'] + p['gs']) # l/r:↑-
            H[base+3, base+3] = E0 + 0.5*p['DeltaSO'] + 0.5*p['muB']*B*(-p['gv'] - p['gs']) # l/r:↓-
        
        # Hopping terms between dots (conserving spin and valley)
        for s in range(4):  # 4 spin/valley combinations
            H[s, s+4] = p['t']
            H[s+4, s] = p['t'].conjugate()  # Hermiticity
            
        return H
    
    def _build_interaction_tensor(self):
        """Builds the U_{hjkm} tensor according to Appendix A of the paper."""
        p = self.params
        U = np.zeros((8, 8, 8, 8), dtype=np.complex128)
        
        # Helper function to assign elements with fermionic symmetries
        def set_U(h, j, k, m, value):
            if abs(U[h,j,k,m]) > 0.0:
                pass
            else:
                U[h,j,k,m] = value
            
            if abs(U[j,h,k,m]) > 0.0:
                pass
            else:
                U[j,h,k,m] = -value

            if abs(U[h,j,m,k]) > 0.0:
                pass
            else:
                U[h,j,m,k] = -value
            
            if abs(U[j,h,m,k]) > 0.0:
                pass
            else:
                U[j,h,m,k] = value
        
        # 1. On-site interactions (Appendix A, first lines)
        set_U(0, 1, 1, 0, p['U0'] + p['J']*(p['G_zz'] + p['G_z0'] + p['G_0z']))
        set_U(2, 3, 3, 2, p['U0'] + p['J']*(p['G_zz'] + p['G_z0'] + p['G_0z']))

        set_U(0, 2, 2, 0, p['U0'] + p['J']*(p['G_zz'] - (p['G_z0'] + p['G_0z'])))
        set_U(0, 3, 3, 0,  p['U0'] + p['J']*(p['G_zz'] - (p['G_z0'] + p['G_0z'])))
        set_U(1, 2, 2, 1,  p['U0'] + p['J']*(p['G_zz'] - (p['G_z0'] + p['G_0z'])))
        set_U(1, 3, 3, 1,  p['U0'] + p['J']*(p['G_zz'] - (p['G_z0'] + p['G_0z'])))

        set_U(0, 2, 0, 2, 4*p['J']*p['G_perp'])
        set_U(0, 3, 1, 2, 4*p['J']*p['G_perp'])
        set_U(1, 3, 1, 3, 4*p['J']*p['G_perp'])
        set_U(1, 2, 0, 3, 4*p['J']*p['G_perp'])
        
        # 2. Direct neighbor interaction (U1)
        for i in range(4):
            for j in range(4):
                set_U(i, j+4, j+4, i, p['U1'])
        
        # 3. Inter-site exchange (X)
        set_U(0, 4, 0, 4, p['X'])
        set_U(1, 5, 1, 5, p['X'])
        set_U(2, 6, 2, 6, p['X'])
        set_U(3, 7, 3, 7, p['X'])
        
        # 4. Pair hoppings (P)
        set_U(0, 1, 5, 4, p['P'])
        set_U(0, 2, 6, 4, p['P'])
        set_U(0, 3, 7, 4, p['P'])
        set_U(1, 2, 6, 5, p['P'])
        set_U(1, 3, 7, 5, p['P'])
        set_U(2, 3, 7, 6, p['P'])
        
        # 5. Density-assisted hoppings (A)
        set_U(1, 2, 6, 1, p['A'])
        set_U(1, 3, 7, 1, p['A'])
        set_U(2, 3, 7, 2, p['A'])
        set_U(1, 6, 2, 1, p['A'])
        set_U(0, 1, 5, 0, p['A'])
        set_U(0, 2, 6, 0, p['A'])
        set_U(0, 3, 7, 0, p['A'])
        set_U(0, 5, 1, 0, p['A'])
        set_U(0, 6, 2, 0, p['A'])
        set_U(0, 7, 3, 0, p['A'])
        set_U(1, 4, 0, 1, p['A'])
        set_U(1, 7, 3, 1, p['A'])
        set_U(2, 7, 3, 2, p['A'])
        set_U(2, 4, 0, 2, p['A'])
        set_U(2, 5, 1, 2, p['A'])
        set_U(3, 4, 0, 3, p['A'])
        set_U(3, 6, 2, 3, p['A'])
        set_U(3, 5, 1, 3, p['A'])
        
        return U
    
    def _construct_basis(self):
        """Builds the antymietric basis of 2 particles: c†_a c†_b |0⟩ con a < b"""
        return [ (a, b) for a in range(self.dim_single) for b in range(a+1, self.dim_single) ]
    
    def build_N2_hamiltonian(self, B):
        """Constructs the Hamiltonian for two particles in the antisymmetric subspace."""
        self.tMatrix = self._single_particle_hamiltonian(B)
        dimBasis = len(self.basis)
        H = lil_matrix((dimBasis, dimBasis), dtype=np.complex128)

        # Single-particle contributions
        for (a, b), idx_ab in self.stateToIndex.items():
            for i in range(self.dim_single):
                # t_{i,a} contribute to a |i,b⟩
                if i != b:
                    newState = tuple(sorted((i, b)))
                    sign = (-1)**(a > i)
                    if newState in self.stateToIndex:
                        H[self.stateToIndex[newState], idx_ab] += self.tMatrix[i, a] * sign

                # t_{i,b} contributes to |i,a⟩
                if i != a:
                    newState = tuple(sorted((i, a)))
                    sign = (-1)**(b > i)
                    if newState in self.stateToIndex:
                        H[self.stateToIndex[newState], idx_ab] += self.tMatrix[i, b] * sign

        # Two-particle interaction contributions
        for (a, b), idx_ab in self.stateToIndex.items():
            for (c, d), idx_cd in self.stateToIndex.items():
                H[idx_cd, idx_ab] += 0.5 * self.uTensor[c, d, a, b]

        # Hermitian symmetry
        H = 0.5 * (H + H.getH())

        return csr_matrix(H)

def plot_spectrum_vs_B(params, B_values):
    """Calculates and plots the spectrum as function of magnetic field."""
    system = BLGDQD_Hubbard(params)
    eigenvalues = []
    
    for B in B_values:
        H_2p = system.build_N2_hamiltonian(B)
        
        # Diagonalize (using sparse for efficiency)
        eigvals = eigsh(H_2p, k=16, which='SA', return_eigenvectors=False)
        eigenvalues.append(np.sort(eigvals))
    
    eigenvalues = np.array(eigenvalues).T
    
    plt.figure(figsize=(10, 6))
    for i in range(eigenvalues.shape[0]):
        plt.plot(B_values, eigenvalues[i], lw=1.5, label=f'State {i+1}')
    
    plt.xlabel('Magnetic field B (T)')
    plt.ylabel('Energy (meV)')
    plt.title('Two-particle spectrum vs magnetic field')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def obtain_eigenvectors_at_zero_B(params):
    """Obtains the eigenvectors at zero magnetic field."""
    system = BLGDQD_Hubbard(params)
    H_2p = system.build_N2_hamiltonian(0.0)
    
    # Diagonalize (using sparse for efficiency)
    eigvals, eigvecs = eigsh(H_2p, k=16, which='SA')
    
    # Sort eigenvalues and corresponding eigenvectors
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Convert eigenvectors to the basis states
    basis_states = [system.basis[i] for i in range(system.dim_single)]
    eigenvectors = []
    for vec in eigvecs.T:
        state_vector = {state: vec[system.stateToIndex[state]] for state in system.basis}
        eigenvectors.append(state_vector)
    eigenvectors = np.array(eigenvectors)
    eigvals = np.array(eigvals)
    eigenvectors = np.array([np.array([state_vector[state] for state in system.basis]) for state_vector in eigenvectors])

    # Print relation between eigenvectors and basis states ordered by absolute weight in the contribution
    for i, vec in enumerate(eigenvectors):
        contributionDict = {state: abs(vec[j]) for j, state in enumerate(system.basis)}
        sorted_states = sorted(contributionDict.items(), key=lambda item: item[1], reverse=True)
        print(f"Eigenvector {i}:")
        for state, weight in sorted_states:
            print(f"  {state}: {weight:.4f}")
             
    return eigvals, eigenvectors

# Typical parameters based on the paper
params = {}

# Obtain eigenvectors at zero magnetic field
eigenvalues, eigenvectors = obtain_eigenvectors_at_zero_B(params)
# Calculate and plot spectrum
B_values = np.linspace(0, 1, 100)  # Magnetic field from 0 to 10 T
plot_spectrum_vs_B(params, B_values)