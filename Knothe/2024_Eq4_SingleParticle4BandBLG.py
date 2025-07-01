import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh

# Global Parameters we will assume the particle can only be in the x direction

V0 = 0.0 # confinement potential
Delta0 = 0.1 # modulated gap
gS = 2.0 # spin g-factor
gV = 1.0 # valley g-factor
DeltaSO = 1e-3 # Kane-Mele spin-orbit gap

Bfield = 10.0  # Tesla
position = 0.0
deltaPArray = np.linspace(0.0, 1e-6, 400)
Kpoint = +1
spin = +1

a = 2.46 / np.sqrt(3)  # lattice spacing ~1.42 Ångstrom
muB = 5.788e-5  # eV/Tesla
electron = 1.0 #Natural units
c = 1.0
hbar = 1.0

v = 1.02e6 #10^6 m/s
v3 = 0.12*v
gamma1 = 0.38


def Vpotential(position, dotPosition):
    """It assumes to be"""
    Vconf = 1.0/(np.cosh(np.sqrt((position)**2)))
    return V0*Vconf

def computeAVector(position, Bfield):
    """Returns y componente of the vector potential in symmetric gauge A = (-By/2, Bx/2)"""
    return 0.5 * Bfield * position

def getBasePVector(Bfield, position):
    """Returns y component of p_0 = -e/c * A(r) assumed to be located around Kpoint"""
    A = computeAVector(position, Bfield)
    return -(electron / c) * A

def piFunction(pVector, Kpoint):
    return Kpoint*pVector[0]+1j*pVector[1]


def singleParticle4BandBLGhamiltonian(pVector, Kpoint, spin, Bfield):
    """Eq. 4"""

    Hb = np.zeros((4,4), dtype=np.complex128)
    V = Vpotential(r,xi)
    Delta = DeltaFunction(r,xi)

    Hb[0,0] = V - 0.5*Kpoint*Delta + spin*Kpoint*DeltaSO+spin*gS*muB*Bfield
    Hb[0,1] = Kpoint*v3*piFunction(pVector, Kpoint)
    Hb[0,2] = 0.0 # v4 assumed to be neglectible
    Hb[0,3] = Kpoint*v*np.conj(piFunction(pVector, Kpoint))

    Hb[1,0] = Kpoint*v3*np.conj(piFunction(pVector, Kpoint))
    Hb[1,1] = V + 0.5*Kpoint*Delta + spin*Kpoint*DeltaSO+spin*gS*muB*Bfield
    Hb[1,2] = Kpoint*v*piFunction(pVector, Kpoint)
    Hb[1,3] = 0.0

    Hb[2,0] = 0.0
    Hb[2,1] = Kpoint*v*np.conj(piFunction(pVector, Kpoint))
    Hb[2,2] = V + 0.5*Kpoint*Delta + spin*Kpoint*DeltaSO+spin*gS*muB*Bfield
    Hb[2,3] = gamma1

    Hb[3,0] = Kpoint*v*piFunction(pVector, Kpoint)
    Hb[3,1] = 0.0
    Hb[3,2] = gamma1
    Hb[3,3] = V - 0.5*Kpoint*Delta + spin*Kpoint*DeltaSO+spin*gS*muB*Bfield

    return Hb


baseP = getBasePVector(Bfield, position)
eigenvaluesList = []

for deltaP in deltaPArray:
    pVector = np.array([0.0, baseP+deltaP])
    Hb = singleParticle4BandBLGhamiltonian(pVector,Kpoint, spin, Bfield)
    eigvals = eigvalsh(Hb)
    eigenvaluesList.append(eigvals)

eigenvaluesArray = np.array(eigenvaluesList).T

plt.figure(figsize=(8, 6))
for i, eig in enumerate(eigenvaluesArray):
    plt.plot(deltaPArray, eig, lw=1.5)


plt.grid(True)
plt.tight_layout()
plt.show()


