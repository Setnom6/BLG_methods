import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh

"""
In this script an effective Hamiltonian of the first orbital of a DQD QD for two particles localized in different dots
is taken into account

The basis used is |K up>, |K down>, |K' up>, |K' down> for each dot

Parameters are:

Delta_SO : SO gap which polarized the spin out-of-plane for 0 magnetic field
gs: spin g-factor which should be equal in any orbital and we usually take as gs = 2.0
gv1: valley g-factor which is different for each orbital
DeltaKK': mixing between valleys K and K' (origin needs to be identified)

t0: spin-conserving tunneling
tv: valley-mixing spin-conserving tunneling
tSO: spin-flip tunneling due to extrinsic SO
tValleySO: valley-spin-flip tunneling due to extrinsic SO (more exotic)
"""


# Parameters

gv1 = 15
DeltaKK = 1e-5 #eV
DeltaSO = 1e-4

t0=1.0
tv=0.0
tSO=[0.0, 0.0, 0.0]
tValleySO=[0.0, 0.0, 0.0]

# Magnetic field to iterate

arrayParameters = np.linspace(0.0, 1.0, 300) # in Tesla

# Constants

gs = 2.0
muB = 5.788e-5 #Bohr magneton in eV/T


def singlePartivleFirstOrbitalSingleQDHamiltonian(magneticField):
    """Obtained from overleaf from Hubert"""

    Hb = np.zeros((4, 4), dtype=np.complex128)
    Bz = magneticField[2]
    Bx = np.linalg.norm(magneticField[0:2])

    Hb[0, 0] = 0.5*muB*(gv1*Bz+gs*Bz)+0.5*DeltaSO
    Hb[0, 1] = 0.5*muB*gs*Bx
    Hb[0, 2] = DeltaKK
    #Hb[0, 3] = 0.0

    Hb[1, 0] = 0.5*muB*gs*Bx
    Hb[1, 1] = 0.5*muB*(gv1*Bz-gs*Bz)-0.5*DeltaSO
    #Hb[1, 2] = 0.0
    Hb[1, 3] = DeltaKK

    Hb[2, 0] = DeltaKK
    #Hb[2, 1] = 0.0
    Hb[2, 2] = 0.5*muB*(-gv1*Bz+gs*Bz)-0.5*DeltaSO
    Hb[2, 3] = 0.5*muB*gs*Bx

   # Hb[3, 0] = 0.0
    Hb[3, 1] = DeltaKK
    Hb[3, 2] = 0.5*muB*gs*Bx
    Hb[3, 3] = 0.5*muB*(-gv1*Bz-gs*Bz)+0.5*DeltaSO

    return Hb


def generalTunnelingHamiltonian():
    """Hamiltoniano general de túnel entre estados de un electrón (4x4)"""
    id2 = np.eye(2, dtype=np.complex128)
    sigmaX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigmaY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigmaZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    sigma = [sigmaX, sigmaY, sigmaZ]

    tMatrix = t0 * np.kron(id2, id2) + tv * np.kron(sigmaX, id2)

    for i in range(3):
        tMatrix += tSO[i] * np.kron(id2, sigma[i])
        tMatrix += tValleySO[i] * np.kron(sigmaX, sigma[i])

    return np.array(tMatrix, dtype=np.complex128)


def buildDQDHamiltonian(magneticFieldL, magneticFieldR):
    """16x16 Hamiltonian"""
    hL = singlePartivleFirstOrbitalSingleQDHamiltonian(magneticFieldL)
    hR = singlePartivleFirstOrbitalSingleQDHamiltonian(magneticFieldR)
    tMat = generalTunnelingHamiltonian()

    id4 = np.eye(4)

    # Hamiltonianos locales: cada uno actúa en uno de los electrones
    hLocal = np.kron(hL, id4) + np.kron(id4, hR)

    # Hamiltoniano de túnel en subespacio (1,1)
    hTunnel = np.kron(tMat, tMat.conj().T)

    return np.array(hLocal + hTunnel, dtype=np.complex128)

eigenvaluesList = []

for B in arrayParameters:
    if B>0:
        Bvector = np.array([0.0,0.0,B])
    else:
        Bvector = np.array([B,0.0,0.0])

    H = buildDQDHamiltonian(Bvector, Bvector)
    eigvals = eigvalsh(H)
    eigenvaluesList.append(eigvals)


eigenvaluesArray = np.array(eigenvaluesList).T

plt.figure(figsize=(8, 6))
for i, eig in enumerate(eigenvaluesArray):
    plt.plot(arrayParameters, eig, lw=1.5)


plt.grid(True)
plt.tight_layout()
plt.show()