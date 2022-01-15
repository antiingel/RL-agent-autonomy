
import numpy as np
from BROJA_2PID import pid


def log2(x):
    return 0 if x == 0 else np.log2(x)


def h(p):
    assert np.allclose(np.sum(p), 1, 1.e-4)
    flat_p = p.flatten()
    r = np.array(list(map(log2, flat_p)))
    return -np.sum(flat_p*r)


def h_cond(a_b, b):
    # H(A|B)
    # a_b = joint AB
    # b = B
    return h(a_b) - h(b)


def mi(a, b, a_b):
    # MI(A:B)
    # a = A
    # b = B
    # a_b = joint AB
    return h(a)-h_cond(a_b, b)


def mi_cond(a_c, a_b_c, b_c, c):
    # MI(A:B|C) = H(A|C) - H(A|BC)
    # a_c = joint AC
    # a_b_c = joint ABC
    # b_c = joint BC
    # c = C
    return h_cond(a_c, c) - h_cond(a_b_c, b_c)


def calculate_autonomy(agent_transition_matrix, env_transition_matrix, row, agent_state_count, env_state_count):
    # P(a)
    p_a = np.array([np.sum(row[i*env_state_count: (i+1)*env_state_count]) for i in range(agent_state_count)])
    # P(c)
    p_c = np.array([np.sum(row[i:agent_state_count*env_state_count:env_state_count]) for i in range(env_state_count)])

    # P(a_n, a_{n-1}, c_{n-1})
    p_a_a_c = agent_transition_matrix

    parms = dict()
    parms['max_iters'] = 50
    probs = {}
    for i in range(agent_state_count):
        for j in range(env_state_count):
            for k in range(agent_state_count):
                probs[(k,j,i)] = np.float(p_a_a_c[i*env_state_count+j][k])
    returndict = pid(probs, cone_solver="ECOS", output=0, **parms)

    shared = returndict["SI"]
    synergistic = returndict["CI"]
    unique_y = returndict["UIY"]
    unique_z = returndict["UIZ"]

    # P(a_n, a_{n-1})
    p_a_n_a_n_1 = np.array([np.sum(p_a_a_c[i*env_state_count: (i+1)*env_state_count], axis=0) for i in range(agent_state_count)])
    # P(a_n, c_{n-1})
    p_a_n_c_n_1 = np.array([np.sum(p_a_a_c[i:agent_state_count*env_state_count:env_state_count], axis=0) for i in range(env_state_count)])

    # MI(S_{n+1} : S_n | E_n)
    Am = mi_cond(p_a_n_c_n_1, p_a_a_c, row, p_c)
    # MI(S_{n+1} : S_n)
    As = mi(p_a, p_a, p_a_n_a_n_1)
    # MI(S_{n+1} : E_n)
    NTIC1 = mi(p_a, p_c, p_a_n_c_n_1)
    # MI(S_{n+1} : E_n | S_n)
    NTIC2 = mi_cond(p_a_n_a_n_1, p_a_a_c, row, p_a)

    # MI(S_{n+1} : (S_n, E_n))
    normalising = mi(row, p_a, p_a_a_c)
    return NTIC1, NTIC2, As, Am, normalising, shared, synergistic, unique_y, unique_z
