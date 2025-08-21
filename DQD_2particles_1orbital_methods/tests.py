
        
nsToMeV = 1519.30 

print(1/nsToMeV, "meV/ns")  # Conversion factor from ns to meV
def gammaRelaxation(t1Ns: float) -> float:
        """Return gamma (units of meV) for relaxation L = sqrt(gamma) * O, given T1 in ns."""
        gamma_ns = 1.0 / t1Ns  # gamma en unidades de ns^{-1}
        gamma_meV = gamma_ns / nsToMeV  # Convertir a meV^{-1}
        return gamma_meV
    

def gammaDephasing(t2Ns: float, t1Ns: float) -> float:
    # Calculamos gamma_dephasing en unidades de ns^{-1}
    gamma_dephasing_ns = (1.0 / t2Ns) - (1.0 / (2 * t1Ns))
    
    # Convertimos a meV^{-1}
    gamma_dephasing_meV = gamma_dephasing_ns / nsToMeV
    return gamma_dephasing_meV


dephasing = gammaDephasing(5000, 200) # Dephasing and spin relaxation time in ns
spinRelaxation = gammaRelaxation(200)  # Spin relaxation time in ns


print(f"Dephasing: {dephasing} meV, Spin Relaxation: {spinRelaxation} meV")
import numpy as np
print("Dephasing:", np.sqrt(spinRelaxation), "meV")