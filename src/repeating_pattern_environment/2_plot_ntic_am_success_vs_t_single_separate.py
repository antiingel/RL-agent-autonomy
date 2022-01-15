
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py

n_prbs = 1
seeds = [100]
n_seeds = len(seeds)
n_steps = 633
NTIC1s, NTIC2s, Ass, Ams, NTICs, normalisings, shareds, synergistics, unique_ys, unique_zs = np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps))
successes, success_hiddens = np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps))
NTIC1ns, NTIC2ns, Asns, Amns, NTICns, normalisingns, sharedns, synergisticns, unique_yns, unique_zns = np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps))
success_perturbs, success_hidden_perturbs = np.zeros((n_prbs, n_seeds, n_steps)),np.zeros((n_prbs, n_seeds, n_steps))

n_elem = 0

min1 = -0.1
max1 = 2.5849625007
min2 = -0.1
max2 = 5.6253706828

min1n = -0.1
max1n = 1.1
min2n = -0.1
max2n = 1.1

min_success = 0.5
max_success = 1.05

for seedi, seed in enumerate(seeds):
    for prbi, prb in enumerate([90]):
        prb /= 100

        NTIC1, NTIC2, As, Am, NTIC, normalising, shared, synergistic, unique_y, unique_z = \
            pickle.load(open(os.path.join(os.pardir, os.pardir, "results", "s_" + str(seed) + "_h_" + str(prb) + ".pkl"), "rb"))

        NTIC1 = np.round(np.array(NTIC1), 10)[:-1]
        NTIC2 = np.round(np.array(NTIC2), 10)[:-1]
        As = np.round(np.array(As), 10)[:-1]
        Am = np.round(np.array(Am), 10)[:-1]
        NTIC = np.round(np.array(NTIC), 10)[:-1]
        normalising = np.round(np.array(normalising), 10)[:-1]
        shared = np.round(np.array(shared), 10)[:-1]
        synergistic = np.round(np.array(synergistic), 10)[:-1]
        unique_y = np.round(np.array(unique_y), 10)[:-1]
        unique_z = np.round(np.array(unique_z), 10)[:-1]

        NTIC1s[prbi, seedi] = ((NTIC1[-n_elem:]))
        NTIC2s[prbi, seedi] = ((NTIC2[-n_elem:]))
        Ass[prbi, seedi] = ((As[-n_elem:]))
        Ams[prbi, seedi] = ((Am[-n_elem:]))
        NTICs[prbi, seedi] = ((NTIC[-n_elem:]))
        normalisings[prbi, seedi] = ((normalising[-n_elem:]))
        shareds[prbi, seedi] = ((shared[-n_elem:]))
        synergistics[prbi, seedi] = ((synergistic[-n_elem:]))
        unique_ys[prbi, seedi] = ((unique_y[-n_elem:]))
        unique_zs[prbi, seedi] = ((unique_z[-n_elem:]))

        with h5py.File(os.path.join(os.pardir, os.pardir, "repeating_pattern_data", "autonomy_s" + str(seed) + "_hp" + str(prb) + ".hdf5"), 'r') as h5_file:
            success = h5_file["success"][:-1]
            success_hidden = h5_file["success_hidden"][:-1]
            successes[prbi, seedi] = ((success[-n_elem:]).squeeze())
            success_hiddens[prbi, seedi] = ((success_hidden[-n_elem:]).squeeze())

x = (np.array(list(range(n_steps)))+1)*100

seed = 0
h = 0

plt.subplot(411)
plt.ylim((min1n, np.max(NTICs)+0.1))
plt.plot(x, NTICs[h,seed,:])
plt.plot(x, shareds[h,seed,:])
plt.plot(x, synergistics[h,seed,:])
plt.legend(("NTIC","SI","CI"), loc="upper right")

plt.subplot(412)
plt.ylim((min2n, np.max(Ams)+0.1))
plt.plot(x, Ams[h,seed,:])
plt.plot(x, unique_zs[h,seed,:])
plt.plot(x, synergistics[h,seed,:])
plt.legend(("$A_m$", "UI$(S_n)$", "CI"), loc="upper right")

plt.subplot(413)
plt.ylim((min_success, max_success))
plt.plot(x, successes[h,seed,:])
plt.plot(x, success_hiddens[h,seed,:])
plt.legend(("Overall success", "Hidden success"), loc="lower right")

plt.subplot(414)
plt.ylim((np.min(normalisings)-0.1, np.max(normalisings)+0.1))
plt.plot(x, normalisings[h,seed,:])
plt.legend(("MI$(S_{n+1}:S_n,E_n)$",), loc="upper right")
plt.xlabel("Training step")

plt.show()
