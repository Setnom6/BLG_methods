from src.DQD_2particles_1orbital import DQD_2particles_1orbital, DQDParameters
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh


def scatterColorSelector(eigenvector: np.ndarray) -> float:
    symmetricMask = np.array([1]*12 + [0]*10 + [1]*6)
    antisymmetricMask = np.array([0]*12 + [1]*10 +[0]*6)

    weightS = np.sum(np.abs(eigenvector)**2 * symmetricMask)
    weightAS = np.sum(np.abs(eigenvector)**2 * antisymmetricMask)
    total = weightS + weightAS

    if total == 0:
        return 0
    return (weightAS - weightS) / total


def scalarToRedBlueColor(value: float) -> tuple:
    value = np.clip(value, -1, 1)
    r = (value + 1) / 2
    g = 1 - np.abs(value)
    b = (1 - value) / 2
    return (r, g, b)


def dominantChargeSector(eigenvector: np.ndarray) -> str:
    mask_20 = np.array([1]*6 + [0]*22)
    mask_11 = np.array([0]*6 + [1]*16 + [0]*6)
    mask_02 = np.array([0]*22 + [1]*6)

    w_20 = np.sum(np.abs(eigenvector)**2 * mask_20)
    w_11 = np.sum(np.abs(eigenvector)**2 * mask_11)
    w_02 = np.sum(np.abs(eigenvector)**2 * mask_02)

    total = w_20 + w_11 + w_02
    if total == 0:
        return 'x'

    w_20 /= total
    w_11 /= total
    w_02 /= total

    if w_20 > 0.6:
        return '^'  # triangle (2,0)
    elif w_02 > 0.6:
        return 's'  # square (0,2)
    elif w_11 > 0.6:
        return 'o'  # cicrle (1,1)
    else:
        return 'x'  # undefined


# Parámetros
gOrtho = 1
U0 = 8.5
fixedParameters = {
    DQDParameters.B_FIELD.value: 0.5,
    DQDParameters.B_PARALLEL.value: 0.0,
    DQDParameters.E_I.value: 0.0,
    DQDParameters.T.value: 0.04,
    DQDParameters.DELTA_SO.value: 0.06,
    DQDParameters.DELTA_KK.value: 0.02,
    DQDParameters.T_SOC.value: 0.0,
    DQDParameters.U0.value: U0,
    DQDParameters.U1.value: 1,
    DQDParameters.X.value: 0.02,
    DQDParameters.G_ORTHO.value: gOrtho,
    DQDParameters.G_ZZ.value: 10 * gOrtho,
    DQDParameters.G_Z0.value: 2 * gOrtho / 3,
    DQDParameters.G_0Z.value: 2 * gOrtho / 3,
    DQDParameters.GS.value: 2.0,
    DQDParameters.GV.value: 28.0,
    DQDParameters.A.value: 0.1,
    DQDParameters.P.value: 0.02,
    DQDParameters.J.value: 0.075 / gOrtho,
}

number_of_eigenstates = 28
arrayToPlot = np.linspace(0.0, 1.5 * U0, 100)

dqd = DQD_2particles_1orbital(fixedParameters)
basis = dqd.FSU.basis
preferred_basis = dqd.singlet_triplet_basis

eigvals = np.zeros((len(arrayToPlot), number_of_eigenstates))
rgbColors = np.zeros((len(arrayToPlot), number_of_eigenstates, 3))  # RGB
markers = np.empty((len(arrayToPlot), number_of_eigenstates), dtype='<U2')

for i, eps_i in enumerate(arrayToPlot):
    parameters_to_change = fixedParameters.copy()
    parameters_to_change[DQDParameters.E_I.value] = eps_i
    projectedH = dqd.project_hamiltonian(preferred_basis, parameters_to_change=parameters_to_change)
    eigval, eigv = eigh(projectedH)

    eigvals[i] = eigval[:number_of_eigenstates].real

    for j in range(number_of_eigenstates):
        symmetry = scatterColorSelector(eigv[:, j])
        rgbColors[i, j] = scalarToRedBlueColor(symmetry)
        markers[i, j] = dominantChargeSector(eigv[:, j])


# Gráfico agrupando por tipo de marcador
plt.figure(figsize=(10, 6))
uniqueMarkers = ['o', '^', 's', 'x']
labels = {
    'o': '(1,1) dominate',
    '^': '(2,0) dominate',
    's': '(0,2) dominate',
    'x': 'Undefined'
}
for marker in uniqueMarkers:
    xs, ys, cs = [], [], []
    for i in range(len(arrayToPlot)):
        for j in range(number_of_eigenstates):
            if markers[i, j] == marker:
                xs.append(arrayToPlot[i])
                ys.append(eigvals[i, j] - arrayToPlot[i])
                cs.append(rgbColors[i, j])
    if xs:
        plt.scatter(xs, ys, c=cs, s=15, marker=marker, label=labels[marker])

plt.xlabel('E_i (meV)')
plt.ylabel('Eigenvalue - E_ref (meV)')
plt.title('Eigenvalues: blue = orbitally symmetric, red = orbitally antisymmetric')
plt.grid(True)
plt.legend(title='Dominant charge sector')
plt.tight_layout()
plt.show()

