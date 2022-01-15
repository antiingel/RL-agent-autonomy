
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

normalise = False

min1 = -0.1
max1 = 1.57
min2 = -0.1
max2 = 2.1
min3 = -0.1
max3 = 2.1

for envi,d in enumerate([0.04, 0.1]):
    envi += 1

    NTIC1, NTIC2, As, Am, NTIC, normalising, shared, synergistic, unique_y, unique_z =\
        pickle.load(open(os.path.join(os.pardir, os.pardir, "results", "autonomy" + str(d) + ".pkl"), "rb"))

    NTIC1 = np.round(np.array(NTIC1), 10)
    NTIC2 = np.round(np.array(NTIC2), 10)
    As = np.round(np.array(As), 10)
    Am = np.round(np.array(Am), 10)
    NTIC = np.round(np.array(NTIC), 10)
    normalising = np.round(np.array(normalising), 10)
    shared = np.round(np.array(shared), 10)
    synergistic = np.round(np.array(synergistic), 10)
    unique_y = np.round(np.array(unique_y), 10)
    unique_z = np.round(np.array(unique_z), 10)

    if normalise:
        idx = np.logical_not(np.isclose(normalising, 0))
        NTIC1[idx] /= normalising[idx]
        NTIC2[idx] /= normalising[idx]
        Am[idx] /= normalising[idx]
        As[idx] /= normalising[idx]
        NTIC[idx] /= normalising[idx]
        shared[idx] /= normalising[idx]
        synergistic[idx] /= normalising[idx]
        unique_y[idx] /= normalising[idx]
        unique_z[idx] /= normalising[idx]

    x = list(range(len(NTIC1)))

    plt.subplot(3, 2, envi)
    if envi == 1 or envi == 2:
        plt.title("$d=" + str(d) + "$")
    plt.ylim((min1, max1))
    plt.plot(x, NTIC)
    plt.plot(x, shared)
    plt.plot(x, synergistic)
    if envi == 2:
        plt.legend(("NTIC","SI","CI"), loc="right")
    if envi == 1:
        plt.ylabel("Bits")

    plt.subplot(3, 2, 2+envi)
    plt.ylim((min2, max2))
    plt.plot(x, Am)
    plt.plot(x, unique_z)
    plt.plot(x, synergistic)
    if envi == 2:
        plt.legend(("$A_m$","UI$(S_n)$","CI"), loc="right")
    if envi == 1:
        plt.ylabel("Bits")

    plt.subplot(3, 2, 4+envi)
    plt.ylim((min3, max3))
    plt.plot(x, As)
    plt.plot(x, unique_z)
    plt.plot(x, shared)
    if envi == 2:
        plt.legend(("$A^*$","UI$(S_n)$","SI"), loc="right")
    if envi == 1:
        plt.ylabel("Bits")
    plt.xlabel("Iteration")

plt.show()
