from qutip import *
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

def hamiltonian(detuning, coulomb, hopping, zeemanLeft, zeemanRight):
    N = 4  # Número de modos (L/R up/down)

    def fermionDestroy(i):
        ops = []
        for j in range(N):
            ops.append(destroy(2) if j == i else qeye(2))
        jwSign = tensor([sigmaz()] * i + [qeye(2)] * (N - i)) # https://en.wikipedia.org/wiki/Jordan%E2%80%93Wigner_transformation
        return tensor(ops) * jwSign

    c = [fermionDestroy(i) for i in range(N)]
    cdag = [ci.dag() for ci in c]
    n = [cdag[i] * c[i] for i in range(N)]

    H_detuning = -detuning * (n[0] + n[1]) + detuning * (n[2] + n[3])
    H_coulomb = coulomb * (n[0] * n[1] + n[2] * n[3])
    H_tunnel = hopping * (cdag[2] * c[0] + cdag[0] * c[2] +
                          cdag[3] * c[1] + cdag[1] * c[3])

    hDotL = zeemanLeft[0]*sigmax() + zeemanLeft[1]*sigmay() + zeemanLeft[2]*sigmaz()
    hDotR = zeemanRight[0]*sigmax() + zeemanRight[1]*sigmay() + zeemanRight[2]*sigmaz()

    H_zeeman = sum(
        hDotL[i, j] * cdag[i] * c[j] for i in [0, 1] for j in [0, 1]
    ) + sum(
        hDotR[i, j] * cdag[i+2] * c[j+2] for i in [0, 1] for j in [0, 1]
    )

    H_full = H_detuning + H_coulomb + H_tunnel + H_zeeman

    basis_states = []
    for occ in combinations(range(N), 2):
        state = basis(2, 1) if 0 in occ else basis(2, 0)
        for i in range(1, N):
            s = basis(2, 1) if i in occ else basis(2, 0)
            state = tensor(state, s)
        basis_states.append(state)

    P = sum([ket * ket.dag() for ket in basis_states])
    H_projected = P * H_full * P

    return H_projected

# Parámetros fijos
coulomb = 5.0
hopping = 0.5
zeemanLeft = np.array([0, 0, 1])
zeemanRight = np.array([0, 0, 1])

# Rango de detuning
detuningValues = np.linspace(0, 1.2*coulomb, 200)

# Constantes derivadas
J = 4 * hopping**2 / coulomb
hz = zeemanLeft[2]

# Calcular eigenenergías
eigenvaluesList = []
eigenstatesList = []

for d in detuningValues:
    H = hamiltonian(d, coulomb, hopping, zeemanLeft, zeemanRight)
    eigvals, eigstates = H.eigenstates()
    sortedIndices = np.argsort(eigvals)
    eigvals = np.array(eigvals)[sortedIndices]
    eigstates = [eigstates[i] for i in sortedIndices]
    eigenvaluesList.append(eigvals)
    if np.isclose(d, 0.0):
        eigenstatesAtZero = eigstates
        eigenvaluesAtZero = eigvals

eigenvaluesArray = np.array(eigenvaluesList).T

# Identificación de estados singlete y tripletes en d=0
# Bases relevantes en notación [L↑, L↓, R↑, R↓]
# Singlete: (|↑,↓> - |↓,↑>) / sqrt(2)
# Triplete0: (|↑,↓> + |↓,↑>) / sqrt(2)
# Triplete+: |↑,↑>
# Triplete−: |↓,↓>

# Construcción explícita en la base Fock
def twoFermionState(modes):
    state = [basis(2, 1) if i in modes else basis(2, 0) for i in range(4)]
    return tensor(state)

ketUpDown = twoFermionState([0, 3])
ketDownUp = twoFermionState([1, 2])
ketUpUp = twoFermionState([0, 2])
ketDownDown = twoFermionState([1, 3])

singlet = (ketUpDown - ketDownUp).unit()
triplet0 = (ketUpDown + ketDownUp).unit()
tripletPlus = ketUpUp
tripletMinus = ketDownDown

referenceStates = {
    'Singlet': singlet,
    'Triplet 0': triplet0,
    'Triplet +': tripletPlus,
    'Triplet -': tripletMinus
}

# Calcular fidelidades y asignar etiquetas
labels = [''] * len(eigenvaluesAtZero)
for i, eigvec in enumerate(eigenstatesAtZero):
    for label, refState in referenceStates.items():
        fidelity = abs(eigvec.dag() * refState)**2
        if fidelity > 0.9:
            labels[i] = label

# Graficar
plt.figure(figsize=(8, 6))
for i, eig in enumerate(eigenvaluesArray):
    plt.plot(detuningValues, eig, lw=1.5)
    if labels[i] != '':
        plt.text(0.02, eig[0]+0.1, labels[i], fontsize=10)

plt.axvline(x=coulomb, color='gray', linestyle='--', label='U')
plt.axhline(y=-J, color='red', linestyle='--', label='-J = -4t²/U')
plt.axhline(y=2*hz, color='blue', linestyle='--', label='2hz')
plt.axhline(y=-2*hz, color='blue', linestyle='--', label='-2hz')

plt.xlabel("Detuning")
plt.ylabel("Eigenenergies")
plt.title("Eigenenergies vs Detuning")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
