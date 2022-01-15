
import pickle
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


seeds = [100,200,300,400,500,600,700,800,900,1000]

n_seeds = len(seeds)
n_elem = 633
fig = plt.figure()

for prbi, prb in enumerate(range(10, 100, 10)):
    prb /= 100

    unique_ys, unique_zs = np.zeros((n_seeds, n_elem)), np.zeros((n_seeds, n_elem))
    successs = np.zeros((n_seeds, n_elem))
    success_pms = np.zeros((n_seeds, n_elem))

    for seedi, seed in enumerate(seeds):

        start = 0
        NTIC1, NTIC2, As, Am, NTIC, normalising, shared, synergistic, unique_y, unique_z = \
            pickle.load(open(os.path.join(os.pardir, os.pardir, "results", "s_" + str(seed) + "_h_" + str(prb) + ".pkl"), "rb"))
        unique_y = np.round(np.array(unique_y), 10)[start:-1]
        unique_z = np.round(np.array(unique_z), 10)[start:-1]

        with h5py.File(os.path.join(os.pardir, os.pardir, "repeating_pattern_data", "autonomy_s" + str(seed) + "_hp" + str(prb) + ".hdf5"), 'r') as h5_file:
            success = h5_file["success"][start:-1]
            successs[seedi,:] = success.flatten()
            success_perturb = h5_file["success_perturb"][start:-1]
            success_pms[seedi,:] = success_perturb.flatten()

        unique_ys[seedi,:] = unique_y
        unique_zs[seedi,:] = unique_z

    unique_ys = unique_ys.reshape(-1)
    unique_zs = unique_zs.reshape(-1)
    successs = successs.reshape(-1)
    success_pms = success_pms.reshape(-1)

    y = unique_zs
    x = success_pms
    plt.subplot(3,3,prbi+1)

    plt.title("$h=" + str(prb) + "$")
    res = plt.scatter(x, y, alpha=0.2, marker=".", c=(np.array(list(range(n_elem))*n_seeds)+1)*100, cmap="winter")
    plt.xlim((0.55, 1.05))
    plt.ylim((-0.1, 6.05))
    if prbi > 5:
        plt.xlabel("Success")
    if prbi % 3 == 0:
        plt.ylabel("UI$(S_n)$")

cbar_ax = fig.add_axes([0.91, 0.35, 0.01, 0.3])
fig.colorbar(res, cax=cbar_ax)
plt.show()
