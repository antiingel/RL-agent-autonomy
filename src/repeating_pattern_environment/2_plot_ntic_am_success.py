
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import h5py

n_prbs = 11
seeds = [100,200,300,400,500,600,700,800,900,1000,2100,2200,2300,2400,2500,2600,2700,2800,2900,21000,3100,3200,3300,3400,3500,3600,3700,3800,3900,31000]
n_seeds = len(seeds)
NTIC1s, NTIC2s, Ass, Ams, NTICs, normalisings, shareds, synergistics, unique_ys, unique_zs = np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds))
successes, success_hiddens = np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds))
NTIC1ns, NTIC2ns, Asns, Amns, NTICns, normalisingns, sharedns, synergisticns, unique_yns, unique_zns = np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds))
success_perturbs, success_hidden_perturbs = np.zeros((n_prbs, n_seeds)),np.zeros((n_prbs, n_seeds))

n_elem = 10
x = np.arange(0.0, 1.1, 0.1)

min1 = -0.1
max1 = 2.6
min2 = -0.1
max2 = 5.6

min1n = -0.1
max1n = 1.1
min2n = -0.1
max2n = 1.1


min_success = 0.5
max_success = 1.05

for seedi, seed in enumerate(seeds):
    for prbi, prb in enumerate(range(0, 110, 10)):
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

        NTIC1n = np.round(np.array(NTIC1), 10)
        NTIC2n = np.round(np.array(NTIC2), 10)
        Asn = np.round(np.array(As), 10)
        Amn = np.round(np.array(Am), 10)
        NTICn = np.round(np.array(NTIC), 10)
        sharedn = np.round(np.array(shared), 10)
        synergisticn = np.round(np.array(synergistic), 10)
        unique_yn = np.round(np.array(unique_y), 10)
        unique_zn = np.round(np.array(unique_z), 10)

        NTIC1n[np.where(normalising!=0)] /= normalising[np.where(normalising!=0)]
        NTIC2n[np.where(normalising!=0)] /= normalising[np.where(normalising!=0)]
        Amn[np.where(normalising!=0)] /= normalising[np.where(normalising!=0)]
        Asn[np.where(normalising!=0)] /= normalising[np.where(normalising!=0)]
        NTICn[np.where(normalising!=0)] /= normalising[np.where(normalising!=0)]
        sharedn[np.where(normalising!=0)] /= normalising[np.where(normalising!=0)]
        synergisticn[np.where(normalising!=0)] /= normalising[np.where(normalising!=0)]
        unique_yn[np.where(normalising!=0)] /= normalising[np.where(normalising!=0)]
        unique_zn[np.where(normalising!=0)] /= normalising[np.where(normalising!=0)]

        NTIC1s[prbi, seedi] = (np.mean(NTIC1[-n_elem:]))
        NTIC2s[prbi, seedi] = (np.mean(NTIC2[-n_elem:]))
        Ass[prbi, seedi] = (np.mean(As[-n_elem:]))
        Ams[prbi, seedi] = (np.mean(Am[-n_elem:]))
        NTICs[prbi, seedi] = (np.mean(NTIC[-n_elem:]))
        normalisings[prbi, seedi] = (np.mean(normalising[-n_elem:]))
        shareds[prbi, seedi] = (np.mean(shared[-n_elem:]))
        synergistics[prbi, seedi] = (np.mean(synergistic[-n_elem:]))
        unique_ys[prbi, seedi] = (np.mean(unique_y[-n_elem:]))
        unique_zs[prbi, seedi] = (np.mean(unique_z[-n_elem:]))

        NTIC1ns[prbi, seedi] = (np.mean(NTIC1n[-n_elem:]))
        NTIC2ns[prbi, seedi] = (np.mean(NTIC2n[-n_elem:]))
        Asns[prbi, seedi] = (np.mean(Asn[-n_elem:]))
        Amns[prbi, seedi] = (np.mean(Amn[-n_elem:]))
        NTICns[prbi, seedi] = (np.mean(NTICn[-n_elem:]))
        sharedns[prbi, seedi] = (np.mean(sharedn[-n_elem:]))
        synergisticns[prbi, seedi] = (np.mean(synergisticn[-n_elem:]))
        unique_yns[prbi, seedi] = (np.mean(unique_yn[-n_elem:]))
        unique_zns[prbi, seedi] = (np.mean(unique_zn[-n_elem:]))

        with h5py.File(os.path.join(os.pardir, os.pardir, "repeating_pattern_data", "autonomy_s" + str(seed) + "_hp" + str(prb) + ".hdf5"), 'r') as h5_file:
            success = h5_file["success"][:-1]
            success_hidden = h5_file["success_hidden"][:-1]
            successes[prbi, seedi] = (np.mean(success[-n_elem:]))
            success_hiddens[prbi, seedi] = (np.nanmean(success_hidden[-n_elem:]))

n = NTIC1s.shape[1]
confidence = 0.95
v = scipy.stats.t.ppf((1 + confidence) / 2., n-1)

NTIC1sd = scipy.stats.sem(NTIC1s, axis=1)*v
NTIC2sd = scipy.stats.sem(NTIC2s, axis=1)*v
Assd = scipy.stats.sem(Ass, axis=1)*v
Amsd = scipy.stats.sem(Ams, axis=1)*v
NTICsd = scipy.stats.sem(NTICs, axis=1)*v
normalisingsd = scipy.stats.sem(normalisings, axis=1)*v
sharedsd = scipy.stats.sem(shareds, axis=1)*v
synergisticsd = scipy.stats.sem(synergistics, axis=1)*v
unique_ysd = scipy.stats.sem(unique_ys, axis=1)*v
unique_zsd = scipy.stats.sem(unique_zs, axis=1)*v
successsd = scipy.stats.sem(successes, axis=1)*v
success_hiddensd = scipy.stats.sem(success_hiddens, axis=1)*v

NTIC1s = np.mean(NTIC1s, axis=1)
NTIC2s = np.mean(NTIC2s, axis=1)
Ass = np.mean(Ass, axis=1)
Ams = np.mean(Ams, axis=1)
NTICs = np.mean(NTICs, axis=1)
normalisings = np.mean(normalisings, axis=1)
shareds = np.mean(shareds, axis=1)
synergistics = np.mean(synergistics, axis=1)
unique_ys = np.mean(unique_ys, axis=1)
unique_zs = np.mean(unique_zs, axis=1)
successes = np.mean(successes, axis=1)
success_hiddens = np.mean(success_hiddens, axis=1)

NTIC1nsd = scipy.stats.sem(NTIC1ns, axis=1)*v
NTIC2nsd = scipy.stats.sem(NTIC2ns, axis=1)*v
Asnsd = scipy.stats.sem(Asns, axis=1)*v
Amnsd = scipy.stats.sem(Amns, axis=1)*v
NTICnsd = scipy.stats.sem(NTICns, axis=1)*v
sharednsd = scipy.stats.sem(sharedns, axis=1)*v
synergisticnsd = scipy.stats.sem(synergisticns, axis=1)*v
unique_ynsd = scipy.stats.sem(unique_yns, axis=1)*v
unique_znsd = scipy.stats.sem(unique_zns, axis=1)*v

NTIC1ns = np.mean(NTIC1ns, axis=1)
NTIC2ns = np.mean(NTIC2ns, axis=1)
Asns = np.mean(Asns, axis=1)
Amns = np.mean(Amns, axis=1)
NTICns = np.mean(NTICns, axis=1)
sharedns = np.mean(sharedns, axis=1)
synergisticns = np.mean(synergisticns, axis=1)
unique_yns = np.mean(unique_yns, axis=1)
unique_zns = np.mean(unique_zns, axis=1)


plt.subplot(421)
plt.xticks(x)
plt.ylim((min1n, np.max(Ams+Amsd)+0.1))
plt.errorbar(x, NTICs, yerr=NTICsd, capsize=3)
plt.errorbar(x, shareds, yerr=sharedsd, capsize=3)
plt.errorbar(x, synergistics, yerr=synergisticsd, capsize=3)
plt.legend(("NTIC","SI","CI"))
plt.ylabel("Bits")

plt.subplot(422)
plt.xticks(x)
plt.ylim((min1n, max1n))
plt.errorbar(x, NTICns, yerr=NTICnsd, capsize=3)
plt.errorbar(x, sharedns, yerr=sharednsd, capsize=3)
plt.errorbar(x, synergisticns, yerr=synergisticnsd, capsize=3)
plt.legend(("Normalised NTIC","Normalised SI","Normalised CI"))

plt.subplot(423)
plt.xticks(x)
plt.ylim((min2n, np.max(Ams+Amsd)+0.1))
plt.errorbar(x, Ams, Amsd, capsize=3)
plt.errorbar(x, unique_zs, unique_zsd, capsize=3)
plt.errorbar(x, synergistics, synergisticsd, capsize=3)
plt.legend(("$A_m$", "UI$(S_n)$", "CI"))
plt.ylabel("Bits")

plt.subplot(424)
plt.xticks(x)
plt.ylim((min2n, max2n))
plt.errorbar(x, Amns, Amnsd, capsize=3)
plt.errorbar(x, unique_zns, unique_znsd, capsize=3)
plt.errorbar(x, synergisticns, synergisticnsd, capsize=3)
plt.legend(("Normalised $A_m$", "Normalised UI$(S_n)$", "Normalised CI"))

plt.subplot(425)
plt.ylim((np.min(normalisings-normalisingsd)-0.1, np.max(normalisings+normalisingsd)+0.1))
plt.xticks(x)
plt.errorbar(x, normalisings, normalisingsd, capsize=3)
plt.legend(("MI$(S_{n+1}:S_n,E_n)$",))
plt.xlabel("$h$")
plt.ylabel("Bits")

plt.subplot(426)
plt.ylim((min_success, max_success))
plt.xticks(x)
plt.errorbar(x, successes, successsd, capsize=3)
plt.errorbar(x, success_hiddens, success_hiddensd, capsize=3)
plt.legend(("Overall success", "Hidden success"))
plt.xlabel("$h$")

plt.show()
