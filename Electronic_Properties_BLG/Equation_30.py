import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh

# Global Parameters (Kpoint dependent)

electron = 1.602e-19
U = 1.0
deltaAB = 0.0
DeltaPrime = 0.0
a = 1.42
hbar = 1.0

gamma0 = 3.16
gamma1 = 0.318
gamma3 = 0.38
gamma4 = 0.0


def piFunction(pVector, Kpoint):
    return Kpoint*pVector[0]+1j*pVector[1]

def hamiltonian(pVector, Kpoint):

    Hb = np.zeros((4,4), dtype=np.complex128)
    epsilonA1 = 0.5*(-U+deltaAB)
    epsilonB1 = 0.5*(-U-deltaAB+2*DeltaPrime)
    epsilonA2 = 0.5*(U +deltaAB +2*DeltaPrime)
    epsilonB2 = 0.5*(U-deltaAB)
    v = np.sqrt(3) * a * gamma0 / (2 * hbar)
    v3 = np.sqrt(3) * a * gamma3 / (2 * hbar)
    v4 = np.sqrt(3) * a * gamma4 / (2 * hbar)

    Hb[0,0] = epsilonA1
    Hb[0,1] = v*np.conj(piFunction(pVector, Kpoint))
    Hb[0,2] = -v4*np.conj(piFunction(pVector, Kpoint))
    Hb[0,3] = v3*piFunction(pVector, Kpoint)

    Hb[1,0] = v*piFunction(pVector, Kpoint)
    Hb[1,1] = epsilonB1
    Hb[1,2] = gamma1
    Hb[1,3] = -v4*np.conj(piFunction(pVector, Kpoint))

    Hb[2,0] = -v4*piFunction(pVector, Kpoint)
    Hb[2,1] = gamma1
    Hb[2,2] = epsilonA2
    Hb[2,3] = v*np.conj(piFunction(pVector, Kpoint))

    Hb[3,0] = v3*np.conj(piFunction(pVector, Kpoint))
    Hb[3,1] = -v4*piFunction(pVector, Kpoint)
    Hb[3,2] = v*piFunction(pVector, Kpoint)
    Hb[3,3] = epsilonB2

    return Hb

def energyBands(pVector, Kpoint):
    """Eq 31 where v4 is neglected"""
    v = np.sqrt(3) * a * gamma0 / (2 * hbar)
    v3 = np.sqrt(3) * a * gamma3 / (2 * hbar)

    p = np.linalg.norm(pVector)
    phi = np.arctan(pVector[1]/pVector[0]) if pVector[0] != 0 else 0.0 # Polar angle


    Gamma = (1/4 * np.pow(np.pow(gamma1,2)-np.pow(v3,2)*np.pow(p,2),2)
             + np.pow(v,2)*np.pow(p,2)*(np.pow(gamma1,2)+np.pow(U,2) + np.pow(v3,2)*np.pow(p,2))
             + 2*Kpoint*gamma1*v3*np.pow(v,2)*np.pow(p,3)*np.cos(3*phi))

    epsilon1Squared = np.pow(gamma1,2)/2 + np.pow(U,2)/4 + (np.pow(v,2)+np.pow(v3,2)/2)*np.pow(p,2) + np.pow(-1,1)*np.sqrt(Gamma)
    epsilon2Squared = np.pow(gamma1, 2) / 2 + np.pow(U, 2) / 4 + (np.pow(v, 2) + np.pow(v3, 2) / 2) * np.pow(p,
                                                                                                             2) + np.pow(
        -1, 2) * np.sqrt(Gamma)

    return np.sqrt(epsilon1Squared), np.sqrt(epsilon2Squared)


parameterArray = np.linspace(0, 0.5*a, 400)
eigenvaluesList = []
energyBandsList = []
Kpoint = +1

for pValue in parameterArray:
    pVector = np.array([pValue,0.0])
    Hb = hamiltonian(pVector, Kpoint)
    eigvals = eigvalsh(Hb)
    eigenvaluesList.append(eigvals)
    epsilon1, epsilon2 = energyBands(pVector, Kpoint)
    energyBandsList.append([epsilon1, epsilon2, -epsilon1, -epsilon2])

eigenvaluesArray = np.array(eigenvaluesList).T
energiesArray = np.array(energyBandsList).T

plt.figure(figsize=(8, 6))
for i, eig in enumerate(eigenvaluesArray):
    plt.plot(parameterArray, eig, lw=1.5)

for i, energy in enumerate(energiesArray):
    plt.plot(parameterArray, energy, lw=1.5, linestyle='dashed')

plt.grid(True)
plt.tight_layout()
plt.show()