import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh

"""
In this script an effective Hamiltonian of the first two orbital of a single QD for two particles (electron) is taken into account

The basis used is directly the one of symmetric and antisymmetric single-triplety states given in equations 6 and 8 of Knothe 2022 (same ordering)
There they consider orbitals n and n+1. We will set N=1, so we consider the first two orbitals of the QD.

In this basis the Hamitlonian is diagonal, and the eigenvalues are given by the energies of the two particles in the QD.

Parameters:

Cs11: energy of the 1 single-particle orbital and the screened electron-electron coulmob interaction
Ca12: energy of the orbitally antisymmetric states of two screened interacting electrons in single-particle orbitals 1 and 2
J: specific dot state characteristics, i.e., dot shape, gap, and mode number
Ec(2): effective charge on the dot when two electrons are present, i.e., the charging energy of the dot with two electrons

gperp: scattering inter-valley coupling
gzz: scattering intra-valley coupling
gz0,g0z: current-current coupling

g: spin g factor which should be equal in any orbital and we usually take as g = 2.0
gv1, gv2: valley g-factor which is different for each orbital

DeltaSO: SO gap which polarized the spin out-of-plane for 0 magnetic field

"""

# Parameters

Cs11 = 0.0 # eV
Ca12 = 2e-3 # eV
J = 1e-2 # eV
Ec2 = 0.0 # eV
gperp = 1e-2 # eV
gzz = 1e-3 # eV
g0z = 1e-3 # eV
gz0 = 1e-3 # eV
g = 2.0
gv1 = 35.0
gv2 = 28.0
DeltaSO = 1e-4 # eV

muB = 5.788e-5 # Bohr magneton in eV/T

# Magnetic field to iterate (we consider only the z component for the Zeeman effect)
arrayParameters = np.linspace(0.0, 0.5, 300) # in Tesla

def twoParticleFirstTwoOrbitalsSingleQDHamiltonian(magneticField):
    """Effective Hamiltonian for two particles in the first two orbitals of a single QD"""

    Hb = np.zeros((16, 16), dtype=np.complex128)

    # Diagonal elements for spatially symmetric states (both particles in the first orbital)
    Hb[0, 0] = Cs11 + (gzz + 4*gperp -g0z  - gz0) * J + Ec2
    Hb[1, 1] = Cs11 + (gzz  + g0z + gz0) * J + Ec2 - 2*gv1*muB*magneticField
    Hb[2, 2] = Cs11 + (gzz  + g0z + gz0) * J + Ec2 + 2*gv1*muB*magneticField
    Hb[3, 3] = Cs11 + (gzz - 4*gperp -g0z  - gz0) * J + Ec2 - g*muB*magneticField
    Hb[4, 4] = Cs11 + (gzz - 4*gperp -g0z  - gz0) * J + Ec2
    Hb[5, 5] = Cs11 + (gzz - 4*gperp -g0z  - gz0) * J + Ec2 + g*muB*magneticField

    # Diagonal elements for spatially antisymmetric states (one particle in each orbital)
    Hb[6, 6] = Ca12 - (gv1 + gv2)*muB*magneticField + Ec2 + 2*DeltaSO - g*muB*magneticField
    Hb[7, 7] = Ca12 - (gv1 + gv2)*muB*magneticField + Ec2
    Hb[8, 8] = Ca12 - (gv1 + gv2)*muB*magneticField + Ec2 - 2*DeltaSO + g*muB*magneticField
    Hb[9, 9] = Ca12 - g*muB*magneticField + Ec2
    Hb[10, 10] = Ca12 + Ec2
    Hb[11, 11] = Ca12 + Ec2 + g*muB*magneticField
    Hb[12, 12] = Ca12 + (gv1 + gv2)*muB*magneticField + Ec2 - 2*DeltaSO - g*muB*magneticField
    Hb[13, 13] = Ca12 + (gv1 + gv2)*muB*magneticField + Ec2
    Hb[14, 14] = Ca12 + (gv1 + gv2)*muB*magneticField + Ec2 + 2*DeltaSO + g*muB*magneticField
    Hb[15, 15] = Ca12 + Ec2

    return Hb

eigenvaluesList = []

for B in arrayParameters:
    H = twoParticleFirstTwoOrbitalsSingleQDHamiltonian(B)
    eigvals = []
    for i in range(16):
        eigvals.append(H[i, i])
    eigenvaluesList.append(eigvals)


eigenvaluesArray = np.array(eigenvaluesList).T

plt.figure(figsize=(8, 6))
for i, eig in enumerate(eigenvaluesArray):
    plt.plot(arrayParameters, eig, lw=1.5)


plt.grid(True)
plt.tight_layout()
plt.show()