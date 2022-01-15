
import numpy as np
from grid_environment.transitions import TransitionMatrix
from grid_environment.states import AgentEnvironmentProductState, NotMoveState


class Policy(TransitionMatrix):
    def __init__(self, system_states, action_states):
        TransitionMatrix.__init__(self, system_states, action_states)

    def initialise_not_moving_policy(self):
        for system_state in self.input_states:
            self.set_probability(system_state, self.output_states[-1], 1)

    def initialize_random_policy(self):
        for system_state in self.input_states:
            for action in self.output_states:
                self.set_probability(system_state, action, 0.2)


def setup_policy_iteration(system_states, action_states, system_action_states):
    policy = Policy(system_states, action_states)
    policy_idxs = []
    for i, system_action_state in enumerate(system_action_states):
        system_state = system_action_state.get_system_state()
        highest_probability_action = policy.get_highest_probability_state(system_state)
        action_state = system_action_state.get_action_state()
        if highest_probability_action == action_state:
            policy_idxs.append(i)
    return policy_idxs


def calculate_values(agent_transitions, env_transitions, reward_transitions, gamma, value_function):
    new_values = np.zeros(len(agent_transitions))
    for i in range(len(agent_transitions)):
        p = np.matmul(agent_transitions[i].T, env_transitions[i])
        row = np.multiply(p, reward_transitions[i].T + gamma * value_function.reshape(p.shape))
        new_values[i] = np.sum(row)
    return new_values


def find_highest_probability_actions(policy_transitions, action_state_count, system_state_count):
    highest_probability_actions = np.argmax(policy_transitions, axis=1)
    return highest_probability_actions + [i*action_state_count for i in range(system_state_count)]


def policy_iteration(agent_transitions, env_transitions, reward_transitions, policy_transitions, system_state_count, action_state_count, gamma):
    value_function = np.matrix(np.zeros(system_state_count))
    policy_idxs = find_highest_probability_actions(policy_transitions, action_state_count, system_state_count)
    saved_values = []
    while True:
        current_agent_transitions = np.matrix(agent_transitions[policy_idxs])
        current_env_transitions = np.matrix(env_transitions[policy_idxs])
        current_reward_transitions = np.matrix(reward_transitions[policy_idxs])
        while True:
            delta = 0
            old_values = value_function
            value_function = calculate_values(current_agent_transitions, current_env_transitions, current_reward_transitions, gamma, value_function)
            delta = np.max((delta, np.max(np.abs(old_values-value_function))))
            if delta < 10**-6:
                break
        saved_values.append(list(value_function))
        policy_stable = True
        old_policy = policy_idxs
        values = calculate_values(np.matrix(agent_transitions), np.matrix(env_transitions), np.matrix(reward_transitions), gamma, value_function)
        values = values.reshape((-1, action_state_count))
        values = np.around(values, 8)
        policy_idxs = np.argmax(values, axis=1).T + [i*action_state_count for i in range(system_state_count)]
        if np.any(old_policy != policy_idxs):
            policy_stable = False
        if policy_stable:
            break
    return saved_values


def make_greedy_policy(system_states, action_states, value_function, tol=8):
    value_function.values = np.around(value_function.values, tol)
    result = Policy(system_states, action_states)
    for i, system_state in enumerate(system_states):
        agent_state = system_state.get_agent_state()
        env_state = system_state.get_environment_state()
        values = []
        for j, action_state in enumerate(action_states):
            next_agent_state = agent_state.move(action_state)
            if next_agent_state != agent_state or action_state == NotMoveState():
                next_state = AgentEnvironmentProductState(next_agent_state, env_state)
                values.append(value_function.get_value(next_state))
        max_value = max(values)
        max_count = np.sum(np.array(values) == max_value)
        k = 0
        for j, action_state in enumerate(action_states):
            if agent_state.move(action_state) != agent_state or action_state == NotMoveState():
                if values[k] == max_value:
                    probability = 1.0/max_count
                    result.set_probability(system_state, action_state, probability)
                k += 1
    return result
