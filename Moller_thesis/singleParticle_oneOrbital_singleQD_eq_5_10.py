import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh

"""
In this script an effective Hamiltonian of the first orbital of a single QD for a single particle (electron) is taken into account

The basis used is |K up>, |K down>, |K' up>, |K' down>

Parameters are:

Delta_SO : SO gap which polarized the spin out-of-plane for 0 magnetic field
gs: spin g-factor which should be equal in any orbital and we usually take as gs = 2.0
gv1: valley g-factor which is different for each orbital
DeltaKK': mixing between valleys K and K' (origin needs to be identified)
"""


# Parameters

gv1 = 15
DeltaKK = 1e-5 #eV
DeltaSO = 1e-4

# Magnetic field to iterate

arrayParameters = np.linspace(-2.0, 2.0, 300) # in Tesla

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

eigenvaluesList = []

for B in arrayParameters:
    if B>0:
        Bvector = np.array([0.0,0.0,B])
    else:
        Bvector = np.array([B,0.0,0.0])

    H = singlePartivleFirstOrbitalSingleQDHamiltonian(Bvector)
    eigvals = eigvalsh(H)
    eigenvaluesList.append(eigvals)


eigenvaluesArray = np.array(eigenvaluesList).T

plt.figure(figsize=(8, 6))
for i, eig in enumerate(eigenvaluesArray):
    plt.plot(arrayParameters, eig, lw=1.5)


plt.grid(True)
plt.tight_layout()
plt.show()

