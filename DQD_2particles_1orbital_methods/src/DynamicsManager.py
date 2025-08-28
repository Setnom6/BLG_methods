import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qutip import *
from pymablock import block_diagonalize, operator_to_BlockSeries, series
from src.DQD_2particles_1orbital import DQD_2particles_1orbital, DQDParameters
import numpy as np
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
from detuning_protocol.realistic_signal import *
from src.LindblandOperator import LindbladOperator
import numpy as np
from qutip import Qobj, mesolve
import logging

def setupLogger():
        DM = DynamicsManager({})
        logDir = DM.figuresDir
        os.makedirs(logDir, exist_ok=True)
        logPath = os.path.join(logDir, "log_results.txt")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(logPath),
                logging.StreamHandler()
            ]
        )

class DynamicsManager:

    def __init__(self, fixedParameters):
        self.dqd = DQD_2particles_1orbital()
        self.basis = self.dqd.singlet_triplet_minimal_basis
        self.correspondence = self.dqd.singlet_tirplet_minimal_correspondence
        self.invCorrespondence = {v: k for k, v in self.correspondence.items()}
        self.nsToMeV = 1519.30 # hbar = 6.582x10-25 GeV s -> 1 GeV-1 = 6.582x10-15 s -> 1 ns = 1519.30 meV-1
        self.fixedParameters = fixedParameters
        self.figuresDir = os.path.join(os.getcwd(), "DQD_2particles_1orbital_methods", "figures")

    def simpleTimeEvolution(self, timesNs,  initialState: np.ndarray = None, cutOffN = None, dephasing = None, spinRelaxation = None, runOptions=None):

        initialStateQobj = self.obtainInitialGroundState(cutOffN=cutOffN)
        if initialState is not None:
            initialStateQobj = Qobj(initialState)
        timesMeV = self.nsToMeV * timesNs

        params = deepcopy(self.fixedParameters)
        H_full = self._getProjecteHamiltonian(params)

        collapseOps = []
        if dephasing is not None:
            collapseOps.extend(self._getProjectedDephasingOperators(dephasing))

        if spinRelaxation is not None:
            collapseOps.extend(self._getProjectedSpinRelaxationOperators(spinRelaxation))

        if cutOffN is not None:
            hEff = H_full[:cutOffN, :cutOffN]
            collapseOpsEff = [op[:cutOffN, :cutOffN] for op in collapseOps]
        else:
            hEff, collapseOpsEff = self.schriefferWolff(H_full, collapseOps)
        
        # Convert to Qobj
        hEffQobj = Qobj(hEff)
        collapseOpsEffQObj = [Qobj(op) for op in collapseOpsEff]

        result = mesolve(hEffQobj, initialStateQobj, timesMeV, c_ops=collapseOpsEffQObj, options=runOptions)
        return np.array([state.diag() for state in result.states]) # We keep the populations


    def detuningProtocol(self, tlistNano, eiValues, cutOffN=None, filter=False, dephasing = None, spinRelaxation = None, runOptions=None):
        """
        Executes a detuning protocol bein agnostic to the sweep shape.
        """
        # --- Convert times to meV
        tlist = self.nsToMeV * tlistNano

        # --- Initial state ---
        rho0 = self.obtainInitialGroundState(detuning=eiValues[0], cutOffN=cutOffN)

        if filter:
            # === Apply physical limitations ===

            # 2) Multi-stage low-pass (e.g. 8 MHz at 4K, 25 MHz at 300K)
            eiValues = applyMultiStageLowPass(eiValues, tlistNano, fcListMHz=(8.0, 25.0), perStageOrder=1)

            # 5) DAC quantization (14-bit AWG)
            eiValues = applyQuantization(eiValues, nBits=14)

            # 6) Add Gaussian noise (~2 μeV rms)
            eiValues = addGaussianNoise(eiValues, sigma=0.002)

        # === Precompute effective Hamiltonians ===
        hEffList = []
        for ei in eiValues:
            params = self.fixedParameters.copy()
            params[DQDParameters.E_I.value] = ei
            H_full = self._getProjecteHamiltonian(params)
            if cutOffN is not None:
                H00_eff = H_full[:cutOffN, :cutOffN]
            else:
                H00_eff,_ = self.schriefferWolff(H_full)
            hEffList.append(Qobj(H00_eff))

        def hEffTimeDependent(t, args):
            idx = np.argmin(np.abs(tlist - t))
            return hEffList[idx]
        
        # Get collapse operators which are the same for any detuning
        collapseOps = []
        if dephasing is not None:
            collapseOps.extend(self._getProjectedDephasingOperators(dephasing))

        if spinRelaxation is not None:
            collapseOps.extend(self._getProjectedSpinRelaxationOperators(spinRelaxation))

        if cutOffN is not None:
            collapseOpsEff = [op[:cutOffN, :cutOffN] for op in collapseOps]
        else:
            _, collapseOpsEff = self.schriefferWolff(H_full, collapseOps)
        collapseOpsEffQObj = [Qobj(op) for op in collapseOpsEff]

        # === Solve dynamics ===
        result = mesolve(hEffTimeDependent, rho0, tlist, c_ops=collapseOpsEffQObj, options=runOptions)
        return np.array([state.diag() for state in result.states])
    
    def obtainOriginalProtocolParameters(self, intervalTimes, totalPoints, interactionDetuning=None):
        """
        Original protocol is composed of a first sweep from 0 to the interaction detuning E, 
        then a plateau at the anticrossing point, a second sweep (quick or slow) to 2*E and finally a plateau at 2*E.
        """
        tTotal = sum(intervalTimes)
        nValues = [int(intervalTimes[i] * totalPoints / tTotal) for i in range(4)]
        tlistNano = np.concatenate([
            np.linspace(0, intervalTimes[0], nValues[0], endpoint=False),
            np.linspace(intervalTimes[0], intervalTimes[0] + intervalTimes[1], nValues[1], endpoint=False),
            np.linspace(intervalTimes[0] + intervalTimes[1], intervalTimes[0] + intervalTimes[1] + intervalTimes[2], nValues[2], endpoint=False),
            np.linspace(intervalTimes[0] + intervalTimes[1] + intervalTimes[2], tTotal, nValues[3])
        ])

        if interactionDetuning is None:
            interactionDetuning = self.fixedParameters[DQDParameters.E_I.value]
        eiIntervals = [0.0,
                    interactionDetuning,
                    2.0 * self.fixedParameters[DQDParameters.U0.value]]
        eiValues = np.concatenate([
            np.linspace(eiIntervals[0], eiIntervals[1], nValues[0], endpoint=False),
            np.full(nValues[1], eiIntervals[1]),
            np.linspace(eiIntervals[1], eiIntervals[2], nValues[2]),
            np.full(nValues[3], eiIntervals[2])
        ])

        return tlistNano, eiValues
    
    def obtainInverseProtocolParameters(self, intervalTimes, totalPoints, interactionDetuning=None):
        """
        Inverse protocol is composed of a first sweep from high detuning to the interaction detuning E, 
        then a plateau at the anticrossing point, a second sweep (quick or slow) to 2*E and finally a plateau at 2*E.
        """
        tTotal = sum(intervalTimes)
        nValues = [int(intervalTimes[i] * totalPoints / tTotal) for i in range(4)]
        tlistNano = np.concatenate([
            np.linspace(0, intervalTimes[0], nValues[0], endpoint=False),
            np.linspace(intervalTimes[0], intervalTimes[0] + intervalTimes[1], nValues[1], endpoint=False),
            np.linspace(intervalTimes[0] + intervalTimes[1], intervalTimes[0] + intervalTimes[1] + intervalTimes[2], nValues[2], endpoint=False),
            np.linspace(intervalTimes[0] + intervalTimes[1] + intervalTimes[2], tTotal, nValues[3])
        ])

        if interactionDetuning is None:
            interactionDetuning = self.fixedParameters[DQDParameters.E_I.value]
        eiIntervals = [2.0* self.fixedParameters[DQDParameters.U0.value],
                    interactionDetuning,
                    2.0 * self.fixedParameters[DQDParameters.U0.value]]
        eiValues = np.concatenate([
            np.linspace(eiIntervals[0], eiIntervals[1], nValues[0], endpoint=False),
            np.full(nValues[1], eiIntervals[1]),
            np.linspace(eiIntervals[1], eiIntervals[2], nValues[2]),
            np.full(nValues[3], eiIntervals[2])
        ])

        return tlistNano, eiValues
    
    def obtainHahnEchoParameters(self, expectedPeriod, totalPoints, interactionDetuning=None):
        """
        Generates parameters for a Hahn echo protocol with slopes.

        Protocol:
            1. High detuning for 2*T
            2. Slope down from high to interaction detuning for 1*T
            3. Interaction detuning for 3*T + T/4
            4. Slope up from interaction to high detuning for 1*T
            5. High detuning for 1*T
            6. Slope down from high to interaction detuning for 1*T
            7. Interaction detuning for 3*T + T/2
            8. Slope up from interaction to high detuning for 1*T
            9. High detuning for 5*T
        """

        if interactionDetuning is None:
            interactionDetuning = self.fixedParameters[DQDParameters.E_I.value]

        highDetuning = 2.0 * self.fixedParameters[DQDParameters.U0.value]

        # Blocks as (duration, targetValue, type)
        blocks = [
            (2 * expectedPeriod, highDetuning, "flat"),
            (1 * expectedPeriod, interactionDetuning, "slope"),
            (3 * expectedPeriod + expectedPeriod / 4, interactionDetuning, "flat"),
            (1 * expectedPeriod, highDetuning, "slope"),
            (1 * expectedPeriod, highDetuning, "flat"),
            (1 * expectedPeriod, interactionDetuning, "slope"),
            (3 * expectedPeriod + expectedPeriod / 2, interactionDetuning, "flat"),
            (1 * expectedPeriod, highDetuning, "slope"),
            (5 * expectedPeriod, highDetuning, "flat")
        ]

        # Total time
        tTotal = sum(duration for duration, _, _ in blocks)

        # Distribute points proportionally
        rawValues = np.array([duration for duration, _, _ in blocks]) * totalPoints / tTotal
        nValues = np.floor(rawValues).astype(int)
        diff = totalPoints - np.sum(nValues)
        if diff > 0:
            for i in np.argsort(-(rawValues - nValues))[:diff]:
                nValues[i] += 1

        # Build time list
        tlistNano = np.linspace(0, tTotal, totalPoints, endpoint=False)

        # Build detuning values
        eiValues = []
        currentVal = highDetuning  # start at high detuning

        for (n, (duration, targetVal, kind)) in zip(nValues, blocks):
            if n == 0:
                continue
            if kind == "flat":
                eiValues.append(np.full(n, targetVal))
                currentVal = targetVal
            elif kind == "slope":
                eiValues.append(np.linspace(currentVal, targetVal, n, endpoint=False))
                currentVal = targetVal
            else:
                raise ValueError(f"Unknown block type: {kind}")

        eiValues = np.concatenate(eiValues)

        return tlistNano, eiValues



    
    def getRunOptions(self, atol = 1e-5, rtol = 1e-3, nsteps = 10000):
        """
        Returns the options for the mesolve function.
        """
        return {
            "nsteps": nsteps,
            "atol":atol,
            "rtol": rtol,
            "method": 'bdf'
        }
    
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
    

    def _getProjectedDephasingOperators(self, gamma = 0.01):
        """
        Returns the dephasing operators projected onto the singlet-triplet basis.
        """
        LM = LindbladOperator(self.dqd.FSU)
        listOfOperators = []
        for i in range(8):
            Li = LM.buildDephasingOperator(i)
            Li_proj = np.sqrt(gamma) * self.dqd.project_hamiltonian(self.basis, alternative_operator=Li)
            listOfOperators.append(Li_proj)
        return listOfOperators
    
    def _getProjectedSpinRelaxationOperators(self, gamma = 0.01):
        """
        Returns the spin relaxation operators projected onto the singlet-triplet basis.
        """
        LM = LindbladOperator(self.dqd.FSU)
        listOfOperators = []
        spinTuples = [(1,0), (3,2), (5,4), (7,6)]
        for tuple in spinTuples:
            Li = LM.buildDecoherenceOperator(tuple[0], tuple[1])
            Li_proj = np.sqrt(gamma) * self.dqd.project_hamiltonian(self.basis, alternative_operator=Li)
            listOfOperators.append(Li_proj)
        return listOfOperators
    

    def gammaFromTime(self, t1_ns: float) -> float:
        """Return gamma (in meV) for relaxation operator L = sqrt(gamma) * O, given T1 in ns."""
        gamma_ns = 1.0 / t1_ns  # gamma in units of ns^{-1}
        gamma_meV = gamma_ns / self.nsToMeV  # Convert to meV
        return gamma_meV
    

    def decoherenceTime(self, t2star_ns: float, t1_ns: float) -> float:
        """Return the decoherence time T2 from T1 and T2star."""
        if abs(t1_ns) < 1e-12 or abs(t2star_ns) < 1e-12:
            return 0.0
        return 1.0 / (1.0 / t2star_ns + 1.0 / (2.0 * t1_ns))
        
    def obtainInitialGroundState(self, detuning = None, cutOffN = None):
        parameters = deepcopy(self.fixedParameters)
        parameters[DQDParameters.E_I.value] = 0.0
        if detuning is not None:
            parameters[DQDParameters.E_I.value] = detuning

        H_full = self._getProjecteHamiltonian(parameters)
        if cutOffN is not None:
            hEff = H_full[:cutOffN, :cutOffN]
        else:
            hEff, _ = self.schriefferWolff(H_full)
        hEffQobj = Qobj(hEff)
        _, evecs = hEffQobj.eigenstates()
        psi0 = evecs[0]
        return psi0 * psi0.dag()


    @staticmethod
    def schriefferWolff(H_full, collapseOp = None):
        N0 = 4
        N1 = 6
        subspace_indices = [0]*N0 + [1]*N1
        H0_tot = H_full[:N0+N1, :N0+N1]
        H0 = np.diag(np.diag(H0_tot))
        H1 = H0_tot-H0

        hamiltonian = [H0, H1]

        H_tilde, U, U_adj = block_diagonalize(hamiltonian, subspace_indices=subspace_indices)
        collapseOpReturn = []
        try:
            transformed_H = np.ma.sum(H_tilde[:2, :2, :3], axis=2)
        except:
            transformed_H = np.ma.sum(H_tilde[:2, :2, :2], axis=2)

        if collapseOp is not None:
            for operator in collapseOp:
                operator_proj = operator[:N0+N1, :N0+N1]  # Project the operator onto the subspace
                if np.allclose(np.abs(operator_proj), 0, atol=1e-10):
                    continue
                operator_series = operator_to_BlockSeries(
                        [np.diag(np.diag(operator_proj)), operator_proj-np.diag(np.diag(operator_proj))], hermitian=True, subspace_indices=subspace_indices)
                
                operator_tilde = series.cauchy_dot_product(U_adj, operator_series, U)
                try: 
                    operator_eff = np.ma.sum(operator_tilde[:2, :2, :3], axis=2)
                except:
                    operator_eff = np.ma.sum(operator_tilde[:2, :2, :2], axis=2)

                if np.ma.is_masked(operator_eff) or np.all(operator_eff.mask):
                    continue
                collapseOpReturn.append(operator_eff[0,0])


        return transformed_H[0, 0], collapseOpReturn
    

    @staticmethod
    def completeSchriefferWolff(H_full):
        N0 = 5
        N1 = 6
        N2 = 17
        
        # Subespacio para primera SWT: modos 1 y 2 (se descarta el 0)
        subspace_indicesi = [0]*N1 + [1]*N2
        
        # Copia submatriz relevante (no modificamos H_full)
        Hi = H_full[N0:, N0:].copy()
        H0i = np.diag(np.diag(Hi))
        H1i = Hi - H0i
        hamiltoniani = [H0i, H1i]
        
        H_tildei, _, _ = block_diagonalize(hamiltoniani, subspace_indices=subspace_indicesi)
        
        # Sumar sobre todos los bloques de la tercera dimensión
        transformed_Hi = np.ma.sum(H_tildei[:2, :2, :2], axis=2)
        H11_eff = transformed_Hi[0, 0] 
        
        # Segunda parte: construir la matriz total sobre subespacios 0+1
        H0_tot = H_full[:N0+N1, :N0+N1].copy()  # copia para no modificar H_full
        
        # Reemplazar el bloque correspondiente (modos 1) por H11_eff
        H0_tot[N0:N0+N1, N0:N0+N1] = H11_eff
        
        H0 = np.diag(np.diag(H0_tot))
        H1 = H0_tot - H0
        
        hamiltonian = [H0, H1]
        
        subspace_indices = [0]*N0 + [1]*N1
        
        H_tilde, _, _ = block_diagonalize(hamiltonian, subspace_indices=subspace_indices)
        
        transformed_H = np.ma.sum(H_tilde[:2, :2, :3], axis=2)
        
        return transformed_H[0, 0]



    def obtainAlternativeProtocol2Parameters(self, intervalTimes, totalPoints):
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

        eiIntervals = []
        eiIntervals.append(0.0)
        eiIntervals.append(self.fixedParameters[DQDParameters.E_I.value]*0.98) # The start and end point of the anticrossing can be stretched or ennahced here
        eiIntervals.append(self.fixedParameters[DQDParameters.E_I.value]*1.02)
        eiIntervals.append(2.0 * self.fixedParameters[DQDParameters.U0.value])

        eiValues = np.concatenate([
        np.linspace(eiIntervals[0], eiIntervals[1], nValues[0], endpoint=False),
        np.linspace(eiIntervals[1], eiIntervals[2], nValues[1], endpoint=False),
        np.linspace(eiIntervals[2], eiIntervals[3], nValues[2]),
        np.full(nValues[3], eiIntervals[3])
        ])

        return tlistNano, eiValues
    
    def obtainAlternativeProtocol2Parameters(self, intervalTimes, totalPoints):
        """
        The detuning sweep starts with an slope from 0 to the anticrossing center.
        Then stays in that point for the desired time for rabi oscillations
        After that, it comes back to 0 detuning
        """
        tTotal = sum(intervalTimes)
        nValues = [int(intervalTimes[i]*totalPoints/tTotal) for i in range(4)]

        tlistNano = np.concatenate([
        np.linspace(0, intervalTimes[0], nValues[0], endpoint=False),
        np.linspace(intervalTimes[0], intervalTimes[0] + intervalTimes[1], nValues[1], endpoint=False),
        np.linspace(intervalTimes[0] + intervalTimes[1], intervalTimes[0] + intervalTimes[1] + intervalTimes[2], nValues[2], endpoint=False),
        np.linspace(intervalTimes[0] + intervalTimes[1] + intervalTimes[2], tTotal, nValues[3])
        ])

        eiIntervals = []
        eiIntervals.append(0.0)
        eiIntervals.append(self.fixedParameters[DQDParameters.E_I.value])

        eiValues = np.concatenate([
        np.linspace(eiIntervals[0], eiIntervals[1], nValues[0], endpoint=False),
        np.full(nValues[1], eiIntervals[1]),
        np.linspace(eiIntervals[1], eiIntervals[0], nValues[2]),
        np.full(nValues[3], eiIntervals[0])
        ])

        return tlistNano, eiValues
    

    