import qmeq
import numpy as np

# Quantum dots parameters
params = {
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

nsingle = 8 # Number of single particle states
nparticles = 2 # Number of particles

def single_particle_hamiltonian(B):
        """Single-particle Hamiltonian including magnetic field effects."""
        p = params
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

hsingle = {}
tunnelingH = single_particle_hamiltonian(0.0)  # Tunneling Hamiltonian at B=0
for i in range(nsingle):
    for j in range(i+1, nsingle):
        hsingle[(i, j)] = tunnelingH[i,j]

def build_interaction_tensor():
        """Builds the U_{hjkm} tensor according to Appendix A of the paper."""
        p = params
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


interactionU = build_interaction_tensor()
coulomb = {}
for i in range(nsingle):
    for j in range(i+1, nsingle):
        for k in range(j+1, nsingle):
            for m in range(k+1, nsingle):
                if interactionU[i, j, k, m] != 0.0:
                    coulomb[(i, j, k, m)] = interactionU[i, j, k, m]