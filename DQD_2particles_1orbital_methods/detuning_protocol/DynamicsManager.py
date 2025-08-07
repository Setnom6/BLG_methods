import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qutip import *
from pymablock import block_diagonalize
from src.DQD_2particles_1orbital import DQD_2particles_1orbital, DQDParameters
import numpy as np
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt

class DynamicsManager:

    def __init__(self, fixedParameters):
        self.dqd = DQD_2particles_1orbital()
        self.basis = self.dqd.singlet_triplet_reordered_basis
        self.correspondence = self.dqd.singlet_triplet_reordered_correspondence
        self.invCorrespondence = {v: k for k, v in self.correspondence.items()}
        self.nsToMeV = 1519.30 # hbar = 6.582x10-25 GeV s -> 1 GeV-1 = 6.582x10-15 s -> 1 ns = 1519.30 meV-1
        self.fixedParameters = fixedParameters
        self.figuresDir = os.path.join(os.getcwd(), "DQD_2particles_1orbital_methods", "figures")

    def simpleTimeEvolution(self, timesNs,  initialState: np.ndarray = None, cutOffN = None):
        initialStateQobj = self.obtainInitialGroundState(cutOffN=cutOffN)
        if initialState is not None:
            initialStateQobj = Qobj(initialState)
        timesMeV = self.nsToMeV * timesNs

        params = deepcopy(self.fixedParameters)
        H_full = self._getProjecteHamiltonian(params)

        if cutOffN is not None:
            hEff = H_full[:cutOffN, :cutOffN]
        else:
            hEff = self.schriefferWolff(H_full)
        hEffQobj = Qobj(hEff)

        result = mesolve(hEffQobj, initialStateQobj, timesMeV, c_ops=[])
        return np.array([state.diag() for state in result.states]) # We keep the populations
    

    def detuningProtocol(self, intervalTimes, totalPoints, cutOffN = None):
        """
        The detuning sweep starts with an slope from 0 to the anticrossing center.
        Then stays in that point for the desired time for rabi oscillations
        After that, a new slope (typically quick but can be done in the desired time) leaves the antocrissoing point behind.
        Finally it rests in high detuning to see the final output (blockade or not)
        """
        tTotal = sum(intervalTimes)
        nValues = [int(intervalTimes[i]*totalPoints/tTotal) for i in range(4)]

        tlistNano = np.concatenate([
        np.linspace(0, intervalTimes[0], nValues[0], endpoint=False),
        np.linspace(intervalTimes[0], intervalTimes[0] + intervalTimes[1], nValues[1], endpoint=False),
        np.linspace(intervalTimes[0] + intervalTimes[1], intervalTimes[0] + intervalTimes[1] + intervalTimes[2], nValues[2], endpoint=False),
        np.linspace(intervalTimes[0] + intervalTimes[1] + intervalTimes[2], tTotal, nValues[3])
        ])

        tlist = self.nsToMeV * tlistNano

        eiIntervals = []
        eiIntervals.append(0.0)
        eiIntervals.append(self.fixedParameters[DQDParameters.E_I.value])
        eiIntervals.append(2.0 * self.fixedParameters[DQDParameters.U0.value])

        eiValues = np.concatenate([
        np.linspace(eiIntervals[0], eiIntervals[1], nValues[0], endpoint=False),
        np.full(nValues[1], eiIntervals[1]),
        np.linspace(eiIntervals[1], eiIntervals[2], nValues[2]),
        np.full(nValues[3], eiIntervals[2])
        ])

        rho0 = self.obtainInitialGroundState(cutOffN=cutOffN)

        # === Precompute effective Hamiltonians ===
        hEffList = []
        for ei in eiValues:
            params = self.fixedParameters.copy()
            params[DQDParameters.E_I.value] = ei

            H_full = self._getProjecteHamiltonian(params)
            if cutOffN is not None:
                H00_eff = H_full[:cutOffN, :cutOffN]
            else:
                H00_eff = self.schriefferWolff(H_full)
            H00_eff_qobj = Qobj(H00_eff)
            hEffList.append(H00_eff_qobj)

        def hEffTimeDependent(t, args):
            idx = np.argmin(np.abs(tlist - t))
            return hEffList[idx]
        

        result = mesolve(hEffTimeDependent, rho0, tlist, c_ops=[])
        return np.array([state.diag() for state in result.states]), tlistNano, eiValues
    

    def detuningProtocolAlternative(self, intervalTimes, totalPoints, cutOffN = None):
        """
        Same protocol as the original but the anticrossing region is not a plateao but an slowly increasing function.
        It allows to leave the anticrossing point smoothly.
        """
        tTotal = sum(intervalTimes)
        nValues = [int(intervalTimes[i]*totalPoints/tTotal) for i in range(4)]

        tlistNano = np.concatenate([
        np.linspace(0, intervalTimes[0], nValues[0], endpoint=False),
        np.linspace(intervalTimes[0], intervalTimes[0] + intervalTimes[1], nValues[1], endpoint=False),
        np.linspace(intervalTimes[0] + intervalTimes[1], intervalTimes[0] + intervalTimes[1] + intervalTimes[2], nValues[2], endpoint=False),
        np.linspace(intervalTimes[0] + intervalTimes[1] + intervalTimes[2], tTotal, nValues[3])
        ])

        tlist = self.nsToMeV * tlistNano

        eiIntervals = []
        eiIntervals.append(0.0)
        eiIntervals.append(self.fixedParameters[DQDParameters.E_I.value]*0.95) # The start and end point of the anticrossing can be stretched or ennahced here
        eiIntervals.append(self.fixedParameters[DQDParameters.E_I.value]*1.05)
        eiIntervals.append(2.0 * self.fixedParameters[DQDParameters.U0.value])

        eiValues = np.concatenate([
        np.linspace(eiIntervals[0], eiIntervals[1], nValues[0], endpoint=False),
        np.linspace(eiIntervals[1], eiIntervals[2], nValues[1], endpoint=False),
        np.linspace(eiIntervals[2], eiIntervals[3], nValues[2]),
        np.full(nValues[3], eiIntervals[3])
        ])

        rho0 = self.obtainInitialGroundState(cutOffN=cutOffN)

        # === Precompute effective Hamiltonians ===
        hEffList = []
        for ei in eiValues:
            params = self.fixedParameters.copy()
            params[DQDParameters.E_I.value] = ei

            H_full = self._getProjecteHamiltonian(params)
            if cutOffN is not None:
                H00_eff = H_full[:cutOffN, :cutOffN]
            else:
                H00_eff = self.schriefferWolff(H_full)
            H00_eff_qobj = Qobj(H00_eff)
            hEffList.append(H00_eff_qobj)

        def hEffTimeDependent(t, args):
            idx = np.argmin(np.abs(tlist - t))
            return hEffList[idx]
        

        result = mesolve(hEffTimeDependent, rho0, tlist, c_ops=[])
        return np.array([state.diag() for state in result.states]), tlistNano, eiValues
    
    def getCurrent(self, populations):
        I_t = []
        for population in populations:
            totalPopulation = (
                population[self.invCorrespondence["LL,S,T-"]]
                + population[self.invCorrespondence["LR,S,T-"]]
                + population[self.invCorrespondence["LR,T+,T-"]]
                + population[self.invCorrespondence["LR,T0,T-"]]
                + population[self.invCorrespondence["LR,T-,T-"]]
            )

            I = (
                population[self.invCorrespondence["LL,S,T-"]]
                + population[self.invCorrespondence["LR,S,T-"]]
            ) / totalPopulation

            I_t.append(I.real)

        return np.array(I_t)
    
    
    def saveResults(self, populations=None, times=None, name=""):
        os.makedirs(self.figuresDir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Save fig
        figPath = os.path.join(self.figuresDir, f"{name}_{timestamp}.png")
        plt.savefig(figPath)
        print(f"Figure saved at: {figPath}")

        # Save params
        paramPath = os.path.join(self.figuresDir, f"parameters_{name}_{timestamp}.txt")
        with open(paramPath, 'w') as f:
            for key, value in self.fixedParameters.items():
                f.write(f"{key}: {value}\n")
        print(f"Parameters saved at: {paramPath}")

        # Save data
        if populations is not None:
            if times is not None:
                data = np.column_stack((times, populations))
                nPops = populations.shape[1]
                header = "time\t" + "\t".join([f"pop{i+1}" for i in range(nPops)])
                dataPath = os.path.join(self.figuresDir, f"populations_with_times_{name}_{timestamp}.txt")
                np.savetxt(dataPath, data, fmt="%.6f", delimiter="\t", header=header, comments="")
                print(f"Populations with times saved at: {dataPath}")
                # To retrieve this kind of data: data = np.loadtxt("populations_with_times_test_2025-08-07_15-42-00.txt", skiprows=1); times = data[:, 0]; populations = data[:, 1:]
            else:
                dataPath = os.path.join(self.figuresDir, f"populations_{name}_{timestamp}.txt")
                np.savetxt(dataPath, populations, fmt="%.6f", delimiter="\t")
                print(f"Populations saved at: {dataPath}")
                # To retrieve this kind of data: populations = np.loadtxt("populations_test_2025-08-07_15-42-00.txt") -> (nTimes, nLevels)

    def _getProjecteHamiltonian(self, parameters):
        return self.dqd.project_hamiltonian(self.basis, parameters_to_change=parameters)
    
    def obtainInitialGroundState(self, detuning = None, cutOffN = None):
        parameters = deepcopy(self.fixedParameters)
        parameters[DQDParameters.E_I.value] = 0.0
        if detuning is not None:
            parameters[DQDParameters.E_I.value] = detuning

        H_full = self._getProjecteHamiltonian(parameters)
        if cutOffN is not None:
            hEff = H_full[:cutOffN, :cutOffN]
        else:
            hEff = self.schriefferWolff(H_full)
        hEffQobj = Qobj(hEff)
        _, evecs = hEffQobj.eigenstates()
        psi0 = evecs[0]
        return psi0 * psi0.dag()


    @staticmethod
    def schriefferWolff(H_full):
        N0 = 5
        N1 = 6
        subspace_indices = [0]*N0 + [1]*N1
        H0_tot = H_full[:N0+N1, :N0+N1]
        H0 = np.diag(np.diag(H0_tot))
        H1 = H0_tot-H0

        hamiltonian = [H0, H1]

        H_tilde, _, _ = block_diagonalize(hamiltonian, subspace_indices=subspace_indices)

        transformed_H = np.ma.sum(H_tilde[:2, :2, :3], axis=2)
        return transformed_H[0, 0] 