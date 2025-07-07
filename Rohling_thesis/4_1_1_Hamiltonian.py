
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh


"""
Hamiltonian for a two particles DQD allowing for assymetric charge configurations (2,0) or (0,2)
A schrieffer-Wolf transformation is carried out (only valid for small detunings)

Parameters:

hv: valley zeeman splitting
hs: spin zeeman splitting
epsilon: detuning between dots (eL-eR)
t: hopping parameter
U: Coulomb repulsion for same occupation of the dots

"""

bArray = np.linspace(0,0.001,300)
eigenvaluesList = []
for Bvalue in bArray:
    # Parameters

    hv = 30*Bvalue
    hs = 2.0*Bvalue
    epsilon = 0.001
    t = 0.005
    U = 0.005

    # Matrix elements

    A = np.zeros((7,))
    B = np.zeros((7,))
    C = np.zeros((7,))

    A[1] = hv
    A[2] = hv
    A[3] = hs
    A[4] = hs
    A[5] = hv+hs
    A[6] = hv-hs

    B[1] = epsilon + hs
    B[2] = epsilon - hs
    B[3] = epsilon + hv
    B[4] = epsilon - hv
    B[5] = epsilon
    B[6] = epsilon

    C[1] = 2*hs
    C[2] = -2*hs
    C[3] = 2*hv
    C[4] = -2*hv

    # Effective Matrix elements

    Jtilde = np.zeros((7,))
    Atilde = np.zeros((7,))
    HSplitted = np.zeros((7,2,2))

    for k in range(7):
        Jtilde[k] = (4*t**2 * U * (U**2 - B[k]**2 - A[k]**2) /
                     (U**4 + B[k]**4 + A[k]**4 - 2*U**2 * B[k]**2 - 2*U**2 * A[k]**2 - 2*A[k]**2*B[k]**2))
        Atilde[k] = A[k]*(1-(Jtilde[k]*(U**2 + B[k]**2 - A[k]**2)/
            2*U*(U**2 - B[k]**2 - A[k]**2)))
        HSplitted[k] = np.array([[0, Atilde[k]],[Atilde[k], Jtilde[k]]]) + np.array([[C[k],0],[0,C[k]]])

    H = np.zeros((14,14))
    for k in range(7):
        H[2*k,2*k] = HSplitted[k][0,0]
        H[2*k,2*k+1] = HSplitted[k][0,1]
        H[2*k+1,2*k] = HSplitted[k][1,0]
        H[2*k+1,2*k+1] = HSplitted[k][1,1]

    eigvals = eigvalsh(H)
    eigenvaluesList.append(eigvals)

eigenvaluesArray = np.array(eigenvaluesList).T

plt.figure(figsize=(8, 6))
for i, eig in enumerate(eigenvaluesArray):
    plt.plot(bArray, eig, lw=1.5)


plt.grid(True)
plt.tight_layout()
plt.show()

