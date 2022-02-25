
import h5py
import numpy as np
from BROJA_2PID import pid
from information_theory import mi, mi_cond
import pickle
import os

n_bins = 15
seeds = [100,200,300,400,500,600,700,800,900,1000,2100,2200,2300,2400,2500,2600,2700,2800,2900,21000,3100,3200,3300,3400,3500,3600,3700,3800,3900,31000]

for seed in seeds:

    for h in range(0, 11):
        file_name = os.path.join(os.pardir, os.pardir, "repeating_pattern_data", "autonomy_s" + str(seed) + "_hp" + str(h / 10.) + ".hdf5")

        with h5py.File(file_name, 'r') as h5_file:

            agent = (np.array(h5_file["memory"]))
            # (634, 20, 30, 32)

            next_agent = agent[:,:,1:,:]
            agent = agent[:,:,:-1,:]
            observation = np.array(h5_file["state"]).squeeze()

            success = (np.array(h5_file["success"])).squeeze()
            success_hidden = (np.array(h5_file["success_hidden"])).squeeze()

            minv = np.min(agent)
            maxv = np.max(agent)

            n_edges = n_bins + 1
            n_memory = agent.shape[-1]
            n_env_states = 6
            edges = np.linspace(minv, maxv, n_edges)
            binned_agent = np.digitize(agent, edges)
            binned_next_agent = np.digitize(next_agent, edges)

            for i in range(observation.shape[-1]):
                zero = np.where(observation[:, :, i] == 0)
                non_zero = np.where(observation[:, :, i] != 0)
                observation[:, :, i][zero] = 3 + (i % 3)
                observation[:, :, i][non_zero] = i % 3
            observation = observation.astype(np.int)

            NTIC1s = []
            NTIC2s = []
            Ass = []
            Ams = []
            NTIC = []
            normalisings = []
            shareds = []
            synergistics = []
            unique_ys = []
            unique_zs = []

            for t in range(agent.shape[0]):
                probs_naea = {}
                probs_a = {}
                probs_e = {}
                probs_naa = {}
                probs_nae = {}
                probs_ae = {}
                probs_na = {}
                n_elements = np.prod(binned_agent.shape[1:3])
                for i in range(binned_agent.shape[1]):
                    for j in range(binned_agent.shape[2]):
                        idx = (tuple(binned_next_agent[t, i, j, :]), observation[t, i, j], tuple(binned_agent[t, i, j, :]))
                        if idx in probs_naea:
                            probs_naea[idx] += float(1. / n_elements)
                        else:
                            probs_naea[idx] = float(1. / n_elements)

                        idx = tuple(binned_agent[t, i, j, :])
                        if idx in probs_a:
                            probs_a[idx] += float(1. / n_elements)
                        else:
                            probs_a[idx] = float(1. / n_elements)

                        idx = tuple(binned_next_agent[t, i, j, :])
                        if idx in probs_na:
                            probs_na[idx] += float(1. / n_elements)
                        else:
                            probs_na[idx] = float(1. / n_elements)

                        idx = observation[t, i, j]
                        if idx in probs_e:
                            probs_e[idx] += float(1. / n_elements)
                        else:
                            probs_e[idx] = float(1. / n_elements)

                        idx = (tuple(binned_next_agent[t, i, j, :]), tuple(binned_agent[t, i, j, :]))
                        if idx in probs_naa:
                            probs_naa[idx] += float(1. / n_elements)
                        else:
                            probs_naa[idx] = float(1. / n_elements)

                        idx = (tuple(binned_next_agent[t, i, j, :]), observation[t, i, j])
                        if idx in probs_nae:
                            probs_nae[idx] += float(1. / n_elements)
                        else:
                            probs_nae[idx] = float(1. / n_elements)

                        idx = (tuple(binned_agent[t, i, j, :]), observation[t, i, j])
                        if idx in probs_ae:
                            probs_ae[idx] += float(1. / n_elements)
                        else:
                            probs_ae[idx] = float(1. / n_elements)

                parms = dict()
                parms['max_iters'] = 50
                returndict = pid(probs_naea, cone_solver="ECOS", output=0, **parms)
                shared = returndict["SI"]
                synergistic = returndict["CI"]
                unique_y = returndict["UIY"]
                unique_z = returndict["UIZ"]

                probs_a = np.array(list(probs_a.values()))
                probs_e = np.array(list(probs_e.values()))
                probs_ae = np.array(list(probs_ae.values()))
                probs_na = np.array(list(probs_na.values()))
                probs_naa = np.array(list(probs_naa.values()))
                probs_nae = np.array(list(probs_nae.values()))
                probs_naea = np.array(list(probs_naea.values()))
                # MI(S_{n+1} : S_n | E_n)
                Am = mi_cond(probs_nae, probs_naea, probs_ae, probs_e)
                # MI(S_{n+1} : S_n)
                As = mi(probs_na, probs_a, probs_naa)
                # MI(S_{n+1} : E_n)
                NTIC1 = mi(probs_na, probs_e, probs_nae)
                # MI(S_{n+1} : E_n | S_n)
                NTIC2 = mi_cond(probs_naa, probs_naea, probs_ae, probs_a)

                # MI(S_{n+1} : (S_n, E_n))
                normalising = mi(probs_ae, probs_na, probs_naea)

                result = NTIC1, NTIC2, As, Am, normalising, shared, synergistic, unique_y, unique_z
                NTIC1s.append(result[0])
                NTIC2s.append(result[1])
                Ass.append(result[2])
                Ams.append(result[3])
                NTIC.append(result[0] - result[1])
                normalisings.append(result[4])
                shareds.append(result[5])
                synergistics.append(result[6])
                unique_ys.append(result[7])
                unique_zs.append(result[8])

            pickle.dump((NTIC1s, NTIC2s, Ass, Ams, NTIC, normalisings, shareds, synergistics, unique_ys, unique_zs),
                        open(os.path.join(os.pardir, os.pardir, "results", "s_" + str(seed) + "_h_" + str(h / 10.) + ".pkl"), "wb"))
