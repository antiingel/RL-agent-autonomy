
from grid_environment.states import *
from grid_environment.transitions import *
from grid_environment.state_merging import merge_states
from information_theory import *

import pickle
import os

# d = 0.1  # if you are in the opposite corner, stay there, otherwise go for the food
# d = 0.04  # always go for the food

for d in [0.1, 0.04]:

    food_transition_probabilities = [
        [1-d, 0, 0, 0, d],
        [0, 1-d, 0, 0, d],
        [0, 0, 1-d, 0, d],
        [0, 0, 0, 1-d, d],
        [0.2, 0.2, 0.2, 0.2, 0.2],
    ]
    after_eating_probabilities = [
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
    ]

    all_values = pickle.load(open(os.path.join(os.pardir, os.pardir, "results", "values" + str(d) + ".pkl"), "rb"))
    all_values = [[0]*len(all_values[0])] + all_values

    n_row = 5
    n_col = 5


    agent_states = make_agent_states(n_row, n_col)
    env_states = make_env_states(n_row, n_col)
    action_states = make_agent_action_states()
    system_states = make_system_states(agent_states, env_states)
    system_action_states = make_system_action_product_states(system_states, action_states)
    agent_transitions = make_agent_transitions(system_action_states, agent_states)
    corner_states = [AgentState(0, 0), AgentState(0, 4), AgentState(4, 0), AgentState(4, 4)]


    env_transitions = make_env_transitions(system_states, env_states, food_transition_probabilities, after_eating_probabilities)
    env_transitions = augment_env_transitions_with_actions(system_action_states, env_states, env_transitions, action_states)

    NTIC1s = []
    NTIC2s = []
    Ass = []
    Ams = []
    NTIC = []
    normalising = []
    shared = []
    synergistic = []
    unique_y = []
    unique_z = []
    for idx, values in enumerate(all_values):

        agent_destination, env_destination, limit_destination = merge_states(
            values, system_states, action_states, agent_states, env_states, corner_states, agent_transitions,
            env_transitions
        )
        result = calculate_autonomy(
            agent_destination,
            env_destination,
            limit_destination,
            agent_destination.shape[1],
            env_destination.shape[1]
        )
        NTIC1s.append(result[0])
        NTIC2s.append(result[1])
        Ass.append(result[2])
        Ams.append(result[3])
        NTIC.append(result[0] - result[1])
        normalising.append(result[4])
        shared.append(result[5])
        synergistic.append(result[6])
        unique_y.append(result[7])
        unique_z.append(result[8])

    pickle.dump((NTIC1s, NTIC2s, Ass, Ams, NTIC, normalising, shared, synergistic, unique_y, unique_z),
                open(os.path.join(os.pardir, os.pardir, "results", "autonomy" + str(d) + ".pkl"), "wb"))
