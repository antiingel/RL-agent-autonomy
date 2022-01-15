
import numpy as np
from grid_environment.states import SystemActionState


class Transitions:
    def __init__(self, input_states, output_states):
        self.input_states = input_states
        self.output_states = output_states
        self.values = np.zeros((len(input_states), len(output_states)))

    def set_value(self, input_state, output_state, value):
        i = self.input_states.index(input_state)
        j = self.output_states.index(output_state)
        self.values[i][j] = value

    def set_values(self, input_state, values):
        for i, value in enumerate(values):
            self.values[self.input_states.index(input_state)][i] = value

    def get_value_given_state(self, input_state):
        return self.values[self.input_states.index(input_state)]

    def get_value(self, input_state, output_state):
        return self.values[self.input_states.index(input_state)][self.output_states.index(output_state)]

    def get_highest_value_state(self, input_state):
        return self.output_states[np.argmax(self.values[self.input_states.index(input_state)])]

    def get_non_zero_states(self, input_state):
        idxs = np.argwhere(np.logical_not(np.isclose(self.values[self.input_states.index(input_state)], 0))).T[0]
        return np.array(self.output_states)[idxs]

    def print(self):
        print(self.output_states)
        for input_state, value in zip(self.input_states, self.values):
            print(input_state, list(value))

    def print_non_zero(self):
        for input_state, values in zip(self.input_states, self.values):
            for output_state, value in zip(self.output_states, values):
                if value != 0:
                    print(input_state, output_state, value)

    def print_given_idxs(self, idxs):
        for input_state, values in zip(np.array(self.input_states)[idxs], np.array(self.values)[idxs]):
            for output_state, value in zip(self.output_states, values):
                if value != 0:
                    print(input_state, output_state, value)


class TransitionMatrix(Transitions):
    def __init__(self, input_states, output_states):
        Transitions.__init__(self, input_states, output_states)
        self.transitions = self

    def set_probability(self, input_state, output_state, probability):
        self.transitions.set_value(input_state, output_state, probability)

    def set_probabilities(self, input_state, probabilities):
        self.transitions.set_values(input_state, probabilities)

    def get_probabilities_given_state(self, input_state):
        return self.transitions.get_value_given_state(input_state)

    def get_probability(self, input_state, output_state):
        return self.transitions.get_value(input_state, output_state)

    def get_highest_probability_state(self, input_state):
        return self.transitions.get_highest_value_state(input_state)

    def sample(self, input_state):
        return np.random.choice(self.transitions.output_states, p=self.transitions.values[self.transitions.input_states.index(input_state)])

    def is_probability_distribution(self):
        return np.allclose(self.values.sum(axis=1), 1)


class RewardTransitions(TransitionMatrix):
    def __init__(self, input_states, output_states):
        TransitionMatrix.__init__(self, input_states, output_states)
        self.rewards = Transitions(input_states, output_states)

    def set_reward(self, input_state, output_state, reward):
        self.rewards.set_value(input_state, output_state, reward)

    def get_rewards_given_state(self, input_state):
        return self.rewards.get_value_given_state(input_state)

    def get_reward(self, input_state, output_state):
        return self.rewards.get_value(input_state, output_state)


def make_env_transitions(system_states, env_states, food_transition_probabilities, after_eating_probabilities):
    result = TransitionMatrix(system_states, env_states)
    for system_state in system_states:
        current_env_state = system_state.get_environment_state()
        current_agent_state = system_state.get_agent_state()
        i = env_states.index(current_env_state)
        if current_agent_state == current_env_state:
            for j, next_env_state in enumerate(env_states):
                result.set_probability(system_state, next_env_state, after_eating_probabilities[i][j])
        else:
            for j, next_env_state in enumerate(env_states):
                result.set_probability(system_state, next_env_state, food_transition_probabilities[i][j])
    return result


def augment_env_transitions_with_actions(system_action_states, env_states, env_transitions, action_states):
    result = TransitionMatrix(system_action_states, env_states)
    for system_action_state in system_action_states:
        system_state = system_action_state.get_system_state()
        for env_state in env_states:
            probability = env_transitions.get_probability(system_state, env_state)
            for action_state in action_states:
                result.set_probability(SystemActionState(system_state, action_state), env_state, probability)
    return result


def make_agent_transitions(system_action_states, agent_states):
    result = RewardTransitions(system_action_states, agent_states)
    for system_action_state in system_action_states:
        action_state = system_action_state.get_action_state()
        system_state = system_action_state.get_system_state()
        agent_state = system_state.get_agent_state()
        next_agent_state = agent_state.move(action_state)
        result.set_probability(system_action_state, next_agent_state, 1)
        env_state = system_state.get_environment_state()
        if agent_state == env_state:
            result.set_reward(system_action_state, next_agent_state, 10)
        if agent_state != next_agent_state:
            current = result.get_reward(system_action_state, next_agent_state)
            result.set_reward(system_action_state, next_agent_state, current-1)
    return result


def make_autonomy_agent_transitions(system_states, action_states, policy_values, agent_transitions_values):
    policy_flat = policy_values.flatten().reshape((-1,1))
    ag_ac_given_s = agent_transitions_values*policy_flat
    result = np.array(
        [ag_ac_given_s[i:i+len(action_states), :].sum(axis=0) for i in range(0, len(system_states)*len(action_states), len(action_states))]
    )
    return result


def make_autonomy_env_transitions(system_states, action_states, policy_values, env_transitions_values):
    policy_flat = policy_values.flatten().reshape((-1,1))
    ag_ac_given_s = env_transitions_values*policy_flat
    result = np.array(
        [ag_ac_given_s[i:i+len(action_states), :].sum(axis=0) for i in range(0, len(system_states)*len(action_states), len(action_states))]
    )
    return result


def make_full_transition_matrix(system_states, agent_transitions, env_transitions):
    result = TransitionMatrix(system_states, system_states)
    result.values = np.array([
            [ap*cp for ap in aps for cp in cps] for aps, cps in zip(agent_transitions.values, env_transitions.values)
    ])
    return result
