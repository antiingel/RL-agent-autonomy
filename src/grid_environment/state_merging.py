
import discreteMarkovChain
import scipy.sparse
from numpy.linalg import norm

from grid_environment.transitions import *
from grid_environment.policy import *


class ValueFunction():
    def __init__(self, states):
        self.states = states
        self.values = np.zeros(len(states))

    def set_value(self, state, value):
        self.values[self.states.index(state)] = value

    def get_value(self, state):
        return self.values[self.states.index(state)]


class MyMarkovChain(discreteMarkovChain.markovChain):
    def myPowerMethod(self, tol=1e-8, maxiter=1e5, start=None):
        P = self.getTransitionMatrix().T
        size = P.shape[0]
        if start is None:
            pi = np.ones(size)/size
        else:
            pi = start
        pi1 = np.zeros(size)
        n = norm(pi - pi1, 1)
        i = 0
        while n > tol and i < maxiter:
            pi1 = P.dot(pi)
            pi = P.dot(pi1)
            n = norm(pi - pi1, 1)
            i += 1
        if i >= maxiter:
            print("Maxiter exceeded!")
        self.pi = pi


def get_agent_state_for_destination(env_state, destination, destinations, system_states):
    result = []
    for i, system_state in enumerate(system_states):
        if system_state.get_environment_state() == env_state:
            if destinations[i] == destination:
                result.append(system_state.get_agent_state())
    return result


def merge_states(values, system_states, action_states, agent_states, env_states, corner_states, agent_transitions, env_transitions):
    destination_states = [i for i in range(2 ** len(corner_states))]

    value_function = ValueFunction(system_states)
    value_function.values = values
    policy = make_greedy_policy(system_states, action_states, value_function)
    assert policy.is_probability_distribution()

    autonomy_agent_transitions_values = make_autonomy_agent_transitions(system_states, action_states, policy.values, agent_transitions.values)
    autonomy_agent_transitions = TransitionMatrix(system_states, agent_states)
    autonomy_agent_transitions.values = autonomy_agent_transitions_values
    assert autonomy_agent_transitions.is_probability_distribution()

    autonomy_agent_transitions_full_values = []
    for agent_state in agent_states:
        for food_state in env_states:
            probas = autonomy_agent_transitions.get_probabilities_given_state(AgentEnvironmentProductState(agent_state, food_state))
            probas = [probas[i] if s2 == food_state else 0 for i, s1 in enumerate(agent_states) for s2 in env_states]
            autonomy_agent_transitions_full_values.append(probas)
    autonomy_agent_transitions_full = TransitionMatrix(system_states, system_states)
    autonomy_agent_transitions_full_values = np.array(autonomy_agent_transitions_full_values)
    autonomy_agent_transitions_full.values = autonomy_agent_transitions_full_values

    n_components, labels = scipy.sparse.csgraph.connected_components(autonomy_agent_transitions_full_values)
    n_states = autonomy_agent_transitions_full_values.shape[0]
    system_destination_states = [None for _ in range(n_states)]
    for label in np.unique(labels):
        idxs = np.where(labels == label)[0]
        component = autonomy_agent_transitions_full_values[np.ix_(idxs, idxs)]
        component_mc = MyMarkovChain(component)
        for i in range(len(idxs)):
            limit_temp = np.zeros(n_states)
            start = np.zeros(len(idxs))
            start[i] = 1
            component_mc.myPowerMethod(tol=1e-9, start=start)
            component_limit = np.round(component_mc.pi, 8)
            limit_temp[idxs] = component_limit

            destination_system_states = np.array(system_states)[np.where(limit_temp != 0)]
            destination_state = 0
            for destination_system_state in destination_system_states:
                agent_state = destination_system_state.get_agent_state()
                if agent_state in corner_states:
                    index = corner_states.index(agent_state)
                    destination_state += 2 ** index

            system_destination_states[idxs[i]] = destination_state

    autonomy_env_transitions_values = make_autonomy_env_transitions(system_states, action_states, policy.values, env_transitions.values)
    autonomy_env_transitions = TransitionMatrix(system_states, env_states)
    autonomy_env_transitions.values = autonomy_env_transitions_values

    full_transitions = make_full_transition_matrix(system_states, autonomy_agent_transitions, autonomy_env_transitions)
    assert full_transitions.is_probability_distribution()

    n_components, labels = scipy.sparse.csgraph.connected_components(full_transitions.values)
    n_states = full_transitions.values.shape[0]
    limit = np.zeros(n_states)
    for label in np.unique(labels):
        idxs = np.where(labels == label)[0]
        component = full_transitions.values[np.ix_(idxs, idxs)]
        component_mc = MyMarkovChain(component)
        component_mc.powerMethod(tol=1e-9)
        component_limit = component_mc.pi
        limit[idxs] = component_limit * np.sum(labels == label) / n_states

    assert np.isclose(limit.sum(), 1)
    assert np.allclose(np.matmul(limit, full_transitions.values), limit)

    agent_state_for_destination = {
        (ei, d): get_agent_state_for_destination(e, d, system_destination_states, system_states)
        for ei, e in enumerate(env_states) for d in destination_states
    }

    # P(D',E',D,E)
    full_transitions_destination = [
        [
            sum(
                sum(
                    full_transitions.values[system_states.index(AgentEnvironmentProductState(a, e)),
                                            system_states.index(AgentEnvironmentProductState(ap, ep))] *
                    limit[system_states.index(AgentEnvironmentProductState(a, e))]
                    for a in agent_states if a in agent_state_for_destination[ei, d]
                )
                for ap in agent_states if ap in agent_state_for_destination[epi, dp]
            )
            for dp in destination_states for epi, ep in enumerate(env_states)
        ]
        for d in destination_states for ei, e in enumerate(env_states)
    ]

    full_transitions_destination = np.round(full_transitions_destination, 9)
    # P(D,E)
    limit_destination = np.sum(full_transitions_destination, axis=0)

    # P(D',D,E)
    agent_destination = np.sum(np.array(full_transitions_destination).reshape((80, 16, 5)), axis=2)
    agent_destination = np.round(agent_destination, 9)

    # P(E',D,E)
    env_destination = np.sum(np.array(full_transitions_destination).reshape((80, 16, 5)), axis=1)
    env_destination = np.round(env_destination, 9)
    return agent_destination, env_destination, limit_destination
