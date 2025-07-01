
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh

# Global Parameters

electron = 1.602e-19
U = 1.0
deltaAB = 0.0
a = 1.42
DeltaPrime = 0.022
gamma0 = 3.16
gamma1 = 0.318
gamma3 = 0.38
gamma4 = 0.14


def fFunction(kVector):
    return np.exp(1j*kVector[1]*a/np.sqrt(3))+2*np.exp(1j*kVector[1]*a/(2*np.sqrt(3)))*np.cos(kVector[0]*a/2)

def hamiltonian(kVector):

    Hb = np.zeros((4,4), dtype=np.complex128)
    epsilonA1 = 0.5*(-U+deltaAB)
    epsilonB1 = 0.5*(-U-deltaAB+2*DeltaPrime)
    epsilonA2 = 0.5*(U +deltaAB +2*DeltaPrime)
    epsilonB2 = 0.5*(U-deltaAB)

    Hb[0,0] = epsilonA1
    Hb[0,1] = -gamma0*fFunction(kVector)
    Hb[0,2] = gamma4*fFunction(kVector)
    Hb[0,3] = -gamma3*np.conj(fFunction(kVector))

    Hb[1,0] = -gamma0*np.conj(fFunction(kVector))
    Hb[1,1] = epsilonB1
    Hb[1,2] = gamma1
    Hb[1,3] = gamma4*fFunction(kVector)

    Hb[2,0] = gamma4*np.conj(fFunction(kVector))
    Hb[2,1] = gamma1
    Hb[2,2] = epsilonA2
    Hb[2,3] = -gamma0*fFunction(kVector)

    Hb[3,0] = -gamma3*fFunction(kVector)
    Hb[3,1] = gamma4*np.conj(fFunction(kVector))
    Hb[3,2] = -gamma0*np.conj(fFunction(kVector))
    Hb[3,3] = epsilonB2

    return Hb



parameterArray = np.linspace(-2*np.pi/a, 2*np.pi/a, 400)
eigenvaluesList = []

for k in parameterArray:
    kVector = np.array([k,0.0,0.0])
    Hb = hamiltonian(kVector)
    eigenvaluesList.append(eigvalsh(Hb))

eigenvaluesArray = np.array(eigenvaluesList).T

plt.figure(figsize=(8, 6))
for i, eig in enumerate(eigenvaluesArray):
    plt.plot(parameterArray, eig, lw=1.5)

plt.grid(True)
plt.tight_layout()
plt.show()


