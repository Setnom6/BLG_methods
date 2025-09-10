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

        initialStateQobj = self.obtainInitialGroundState(cutOffN=cutOffN, detuning= self.fixedParameters[DQDParameters.E_I.value], bField = self.fixedParameters[DQDParameters.B_PARALLEL.value])
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


    def detuningProtocol(self, tlistNano, eiValues, cutOffN=None, dephasing = None, spinRelaxation = None, runOptions=None, initialStateDetuning=None, initialStateField=None):
        """
        Executes a detuning protocol bein agnostic to the sweep shape.
        """
        # --- Convert times to meV
        tlist = self.nsToMeV * tlistNano

        # --- Initial state ---
        if initialStateField is None:
            initialStateField = self.fixedParameters[DQDParameters.B_PARALLEL.value]
        if initialStateDetuning is None:
            initialStateDetuning = eiValues[0]
        
        rho0 = self.obtainInitialGroundState(detuning = initialStateDetuning, bField=initialStateField, cutOffN=cutOffN)

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
        collapseOpsEffQObj = self._getAllCollapseOperators(dephasing, spinRelaxation, cutOffN, H_full)

        # === Solve dynamics ===
        result = mesolve(hEffTimeDependent, rho0, tlist, c_ops=collapseOpsEffQObj, options=runOptions)
        return result

    def magneticFieldProtocol(self, tlistNano, bValues, cutOffN=None, dephasing = None, spinRelaxation = None, runOptions=None, initialStateField=None, initialStateDetuning=None):
        """
        Executes a magnetic protocol bein agnostic to the sweep shape.
        """
        # --- Convert times to meV
        tlist = self.nsToMeV * tlistNano

        # --- Initial state ---
        if initialStateField is None:
            initialStateField = bValues[0]
        if initialStateDetuning is None:
            initialStateDetuning = self.fixedParameters[DQDParameters.E_I.value]
        
        rho0 = self.obtainInitialGroundState(detuning = initialStateDetuning, bField=initialStateField, cutOffN=cutOffN)

        # === Precompute effective Hamiltonians ===
        hEffList = []
        for bx in bValues:
            params = self.fixedParameters.copy()
            params[DQDParameters.B_PARALLEL.value] = bx
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
        collapseOpsEffQObj = self._getAllCollapseOperators(dephasing, spinRelaxation, cutOffN, H_full)

        # === Solve dynamics ===
        result = mesolve(hEffTimeDependent, rho0, tlist, c_ops=collapseOpsEffQObj, options=runOptions)
        return result
    
    def combinedProtocol(self, tlistNano, bValues, eiValues, cutOffN=None, dephasing = None, spinRelaxation = None, runOptions=None, initialStateField=None, initialStateDetuning=None):
        """
        Executes a magnetic adn detuning protocol bein agnostic to the sweep shape.
        """
        # --- Convert times to meV
        tlist = self.nsToMeV * tlistNano

        # --- Initial state ---
        if initialStateField is None:
            initialStateField = bValues[0]
        if initialStateDetuning is None:
            initialStateDetuning = eiValues[0]
        
        rho0 = self.obtainInitialGroundState(detuning = initialStateDetuning, bField=initialStateField, cutOffN=cutOffN)

        # === Precompute effective Hamiltonians ===
        hEffList = []
        for index in range(len(bValues)):
            params = self.fixedParameters.copy()
            params[DQDParameters.B_PARALLEL.value] = bValues[index]
            params[DQDParameters.E_I.value] = eiValues[index]
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
        collapseOpsEffQObj = self._getAllCollapseOperators(dephasing, spinRelaxation, cutOffN, H_full)

        # === Solve dynamics ===
        result = mesolve(hEffTimeDependent, rho0, tlist, c_ops=collapseOpsEffQObj, options=runOptions)
        return result

    def buildGenericProtocolParameters(self, listSlopes, totalPoints):
        """
        Build a generic detuning protocol from a list of segments,
        ignoring segments with duration < 1e-10, and distributing
        totalPoints proportionally among the remaining segments.
        
        Args:
            listSlopes: list of lists, each element is [detuningStart, detuningEnd, duration]
            totalPoints: int, total number of points in the concatenated protocol
        
        Returns:
            tlistNano: np.array, concatenated time list
            eiValues: np.array, concatenated detuning values
        """
        # Filter out segments with negligible duration
        filteredSlopes = [seg for seg in listSlopes if seg[2] >= 1e-10]
        
        if not filteredSlopes:
            return np.array([]), np.array([])  # nothing to build

        # Total duration
        totalTime = sum(seg[2] for seg in filteredSlopes)
        
        # Initial number of points per segment (proportional to duration)
        rawPoints = [seg[2] * totalPoints / totalTime for seg in filteredSlopes]
        nValues = [int(rp) for rp in rawPoints]
        
        # Adjust to ensure the sum of points is exactly totalPoints
        pointsDiff = totalPoints - sum(nValues)
        # Distribute remaining points to segments with largest fractional part
        fractionalParts = [rp - int(rp) for rp in rawPoints]
        for idx in sorted(range(len(filteredSlopes)), key=lambda i: fractionalParts[i], reverse=True)[:pointsDiff]:
            nValues[idx] += 1
        
        tlistNano = []
        parameterValues = []
        tAcc = 0.0  # accumulated time
        
        for seg, nPoints in zip(filteredSlopes, nValues):
            tStart = tAcc
            tEnd = tAcc + seg[2]
            
            # Time array for this segment
            if nPoints == 1:
                tSegment = np.array([tStart])
            else:
                tSegment = np.linspace(tStart, tEnd, nPoints, endpoint=False)
            
            # Detuning array for this segment
            detStart, detEnd = seg[0], seg[1]
            if detStart == detEnd:
                eSegment = np.full(nPoints, detStart)
            else:
                eSegment = np.linspace(detStart, detEnd, nPoints, endpoint=False)
            
            tlistNano.append(tSegment)
            parameterValues.append(eSegment)
            
            tAcc = tEnd  # update accumulated time for next segment
        
        # Concatenate all segments
        tlistNano = np.concatenate(tlistNano)
        parameterValues = np.concatenate(parameterValues)
        
        return tlistNano, parameterValues

    
    def densityToSTQubit(self, rho4, iSym, iAnti):
        """
        Project a NxN density matrix onto {|S>,|T>} qubit basis by disregarding internal coherences and maintaining coherences between the two subspaces.
        The indices must be in agreement with the dimension of the density matrix given
        """
        if isinstance(rho4, Qobj):
            rho = rho4.full()
        else:
            rho = np.asarray(rho4, dtype=complex)
        s, t = np.array(iSym), np.array(iAnti)

        rhoSS = np.trace(rho[np.ix_(s, s)])
        rhoTT = np.trace(rho[np.ix_(t, t)])
        rhoST = np.sum(rho[np.ix_(s, t)])
        rhoTS = np.conjugate(rhoST)

        rho2 = np.array([[rhoSS, rhoST],
                        [rhoTS, rhoTT]], dtype=complex)
        rho2 /= np.trace(rho2)
        return rho2
    
    def rho2ToBloch(self, rho2):
        """Return Bloch vector (sx, sy, sz) from a 2x2 density matrix."""
        rho01 = rho2[0, 1]
        sx = 2*np.real(rho01)
        sy = -2*np.imag(rho01)
        sz = np.real(rho2[0, 0] - rho2[1, 1])
        blochVec =  np.array([sx, sy, sz], dtype=float)
        # Clip if norm slightly exceeds 1 (numerical errors / projection artifacts)
        r = np.linalg.norm(blochVec)
        if r > 1.0:
            blochVec = blochVec / r

        return blochVec


    
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
    
    def getCurrent(self, populations, cutOff=None):
        I_t = []

        if cutOff is not None:
            antisymmetricIndices = [
                    self.invCorrespondence['LR,T-,T-'],
                    self.invCorrespondence['LR,T0,T-'],
                    self.invCorrespondence['LR,T-,T0'],
                    self.invCorrespondence['LR,T+,T-'],
                    self.invCorrespondence['LR,T0,T0'],
                    self.invCorrespondence['LR,T+,T0'],
                    self.invCorrespondence['LR,T0,T+'],
                    self.invCorrespondence['LR,T+,T+'],
                    self.invCorrespondence['LR,S,S'],
                    self.invCorrespondence['LR,T-,T+'],
                ]

            symmetricIndices = [i for i in range(0,28) if i not in antisymmetricIndices]

            antisymmetricIndicesCutOff = [i for i in antisymmetricIndices if i < cutOff]
            symmetricIndicesCutOff = [i for i in symmetricIndices if i < cutOff]
            for population in populations:
                totalPopulation = np.sum([population[idx] for idx in antisymmetricIndicesCutOff]) + np.sum([population[idx] for idx in symmetricIndicesCutOff])
                I  = np.sum([population[idx] for idx in symmetricIndicesCutOff])
                I_t.append(I.real)
                
        else:
            for population in populations:
                totalPopulation = (
                    population[self.invCorrespondence["LL,S,T-"]]
                    + population[self.invCorrespondence["LR,S,T-"]]
                    + population[self.invCorrespondence["LR,T0,T-"]]
                    + population[self.invCorrespondence["LR,T-,T-"]]
                )

                I = (
                    population[self.invCorrespondence["LL,S,T-"]]
                    + population[self.invCorrespondence["LR,S,T-"]]
                ) / totalPopulation

                I_t.append(I.real)

        return np.array(I_t)
    
    def getSingletTripletPopulations(self, populations, cutOff=None):
        sumSinglet = []
        sumTriplet = []
        sumTotal = []

        if cutOff is not None:
            antisymmetricIndices = [
                    self.invCorrespondence['LR,T-,T-'],
                    self.invCorrespondence['LR,T0,T-'],
                    self.invCorrespondence['LR,T-,T0'],
                    self.invCorrespondence['LR,T+,T-'],
                    self.invCorrespondence['LR,T0,T0'],
                    self.invCorrespondence['LR,T+,T0'],
                    self.invCorrespondence['LR,T0,T+'],
                    self.invCorrespondence['LR,T+,T+'],
                    self.invCorrespondence['LR,S,S'],
                    self.invCorrespondence['LR,T-,T+'],
                ]

            symmetricIndices = [i for i in range(0,28) if i not in antisymmetricIndices]

            antisymmetricIndicesCutOff = [i for i in antisymmetricIndices if i < cutOff]
            symmetricIndicesCutOff = [i for i in symmetricIndices if i < cutOff]
            for population in populations:
                singletPopulation  = np.sum([population[idx] for idx in symmetricIndicesCutOff])
                tripletPopulation = np.sum([population[idx] for idx in antisymmetricIndicesCutOff])

                sumSinglet.append(singletPopulation.real)
                sumTriplet.append(tripletPopulation.real)
                sumTotal.append(singletPopulation.real + tripletPopulation.real)
                
        else:
            for population in populations:
                tripletPopulation = (
                    population[self.invCorrespondence["LR,T0,T-"]]
                    + population[self.invCorrespondence["LR,T-,T-"]]
                )

                singletPopulation = (
                    population[self.invCorrespondence["LL,S,T-"]]
                    + population[self.invCorrespondence["LR,S,T-"]]
                )

                sumSinglet.append(singletPopulation.real)
                sumTriplet.append(tripletPopulation.real)
                sumTotal.append(singletPopulation.real + tripletPopulation.real)

        return np.array(sumSinglet), np.array(sumTriplet), np.array(sumTotal)


    def computeSingletTripletEnergyDifference(self, eiValues, cutOffN=None, iSym=None, iAnti=None):
        """
        Compute ΔE(t) = E_S(t) - E_T(t) between singlet and triplet subspaces
        by projecting the effective Hamiltonian onto the {|S>,|T>} basis.

        Parameters
        ----------
        eiValues : array_like
            Detuning values (meV) for which to evaluate the Hamiltonian.
        cutOffN : int, optional
            Dimension cutoff for the effective Hamiltonian.
        iSym : list[int], optional
            Indices for the singlet subspace in the cutoff Hamiltonian.
        iAnti : list[int], optional
            Indices for the triplet subspace in the cutoff Hamiltonian.

        Returns
        -------
        deltaE : np.ndarray
            Array of shape (len(eiValues),), with energy differences ΔE at each detuning.
        """
        deltaE = []

        for ei in eiValues:
            # --- Construir Hamiltoniano en este detuning
            params = self.fixedParameters.copy()
            params[DQDParameters.E_I.value] = ei
            H_full = self._getProjecteHamiltonian(params)

            if cutOffN is not None:
                H_eff = H_full[:cutOffN, :cutOffN]
            else:
                H_eff, _ = self.schriefferWolff(H_full)

            # --- Proyección al subespacio {S,T}
            s, t = np.array(iSym), np.array(iAnti)
            HSS = np.trace(H_eff[np.ix_(s, s)])
            HTT = np.trace(H_eff[np.ix_(t, t)])
            HST = np.sum(H_eff[np.ix_(s, t)])
            HTS = np.conjugate(HST)

            H2 = np.array([[HSS, HST],
                        [HTS, HTT]], dtype=complex)

            # Normalizar (opcional: dividir por tamaño del subespacio)
            dimS, dimT = len(s), len(t)
            H2[0, 0] /= dimS if dimS > 0 else 1
            H2[1, 1] /= dimT if dimT > 0 else 1

            # --- Diagonalizar 2x2 proyectado
            evals, _ = np.linalg.eigh(H2)
            deltaE.append(np.diff(evals)[0].real)  # E_high - E_low

        return np.array(deltaE)

    
    
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
    
    def _getAllCollapseOperators(self, dephasing, spinRelaxation, cutOffN, H_full):
        collapseOps = []
        if dephasing is not None:
            collapseOps.extend(self._getProjectedDephasingOperators(dephasing))

        if spinRelaxation is not None:
            collapseOps.extend(self._getProjectedSpinRelaxationOperators(spinRelaxation))

        if cutOffN is not None:
            collapseOpsEff = [op[:cutOffN, :cutOffN] for op in collapseOps]
        else:
            _, collapseOpsEff = self.schriefferWolff(H_full, collapseOps)

        return [Qobj(op) for op in collapseOpsEff]
    

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
        
    def obtainInitialGroundState(self, detuning = None, cutOffN = None, bField = None):
        parameters = deepcopy(self.fixedParameters)
        parameters[DQDParameters.E_I.value] = 0.0
        parameters[DQDParameters.B_PARALLEL.value] = 0.1
        if detuning is not None:
            parameters[DQDParameters.E_I.value] = detuning

        if bField is not None:
            parameters[DQDParameters.B_PARALLEL.value] = bField

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
    

    