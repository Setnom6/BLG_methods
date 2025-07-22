import numpy as np
from src.DQD_2particles_1orbital import DQD_2particles_1orbital

dqd = DQD_2particles_1orbital()

v1 = dqd.singlet_triplet_basis[0]

v = np.array([1/np.sqrt(2)]+[-1/np.sqrt(2)]+[0]*26)

for i in range(28):
        print(dqd.obtain_total_spin(dqd.singlet_triplet_basis[i]))