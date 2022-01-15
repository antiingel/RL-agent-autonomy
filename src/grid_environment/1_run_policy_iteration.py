
from grid_environment.states import *
from grid_environment.transitions import *
from grid_environment.policy import *

import pickle
import os


# d = 0.1  # if you are in the opposite corner, stay there, otherwise go for the food
# d = 0.04  # always go for the food

for d in [0.1, 0.04]:

    n_row = 5
    n_col = 5

    agent_states = make_agent_states(n_row, n_col)
    env_states = make_env_states(n_row, n_col)

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

    action_states = make_agent_action_states()
    system_states = make_system_states(agent_states, env_states)
    system_action_states = make_system_action_product_states(system_states, action_states)
    env_transitions = make_env_transitions(system_states, env_states, food_transition_probabilities, after_eating_probabilities)
    env_transitions = augment_env_transitions_with_actions(system_action_states, env_states, env_transitions, action_states)
    agent_transitions = make_agent_transitions(system_action_states, agent_states)

    assert env_transitions.is_probability_distribution()
    assert agent_transitions.is_probability_distribution()

    policy = Policy(system_states, action_states)
    policy.initialise_not_moving_policy()
    policy_idxs = setup_policy_iteration(system_states, action_states, system_action_states)
    values = policy_iteration(agent_transitions.values, env_transitions.values, agent_transitions.rewards.values, policy.values, len(system_states), len(action_states), 0.9)

    pickle.dump(values, open(os.path.join(os.pardir, os.pardir, "results", "values" + str(d) + ".pkl"), "wb"))
