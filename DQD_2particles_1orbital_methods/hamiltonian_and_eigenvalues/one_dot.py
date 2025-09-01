import numpy as np
import matplotlib.pyplot as plt

bFields = np.linspace(0, 1.5, 100)
eigenvaluesList = []
eigenvectorsList = []

for bField in bFields:
    h = np.zeros((4, 4))

    mub = 5.7883818060e-2  # meV/T
    gs = 2.0
    gv = 28.0
    DeltaSO = 0.066  # meV
    bParallel = 0.0
    gsLeftFactor = 1.0
    gvLeftFactor = 0.66

    spinZeemanLeft = 0.5 * gs * mub * bField * gsLeftFactor
    valleyZeemanLeft = 0.5 * gv * mub * bField * gvLeftFactor
    kaneMele = 0.5 * DeltaSO

    h[0, 0] = kaneMele + valleyZeemanLeft + spinZeemanLeft   # |up K>
    h[1, 1] = -kaneMele + valleyZeemanLeft - spinZeemanLeft  # |down K>
    h[2, 2] = -kaneMele - valleyZeemanLeft + spinZeemanLeft  # |up K'>
    h[3, 3] = kaneMele - valleyZeemanLeft - spinZeemanLeft   # |down K'>

    eigenvalues, eigenvectors = np.linalg.eigh(h)
    eigenvaluesList.append(eigenvalues)
    eigenvectorsList.append(eigenvectors)

eigenvaluesArray = np.array(eigenvaluesList)      # shape (len(bFields),4)
eigenvectorsArray = np.array(eigenvectorsList)    # shape (len(bFields),4,4)

# Colores asignados a cada base
colors = ['red', 'blue', 'green', 'purple']
labels = ['|up K>', '|down K>', "|up K'>", "|down K'>"]

fig, ax = plt.subplots(figsize=(8,6))

for i in range(4):  # índice de eigenvalor
    for idx, bField in enumerate(bFields):
        eigenvec = eigenvectorsArray[idx,:,i]
        overlaps = np.abs(eigenvec)**2
        dominant = np.argmax(overlaps)  # qué base domina
        ax.scatter(bField, eigenvaluesArray[idx,i],
                   color=colors[dominant], s=10)

ax.set_xlabel("Magnetic Field (T)")
ax.set_ylabel("Energy (meV)")
ax.set_title("Eigenvalues colored by dominant basis state")

# construir leyenda dummy
for c, l in zip(colors, labels):
    ax.scatter([], [], color=c, label=l)
ax.legend()

plt.show()