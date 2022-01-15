

class State():
    pass


class ProductState(State):
    def __init__(self, state1, state2):
        State.__init__(self)
        self.state1 = state1
        self.state2 = state2

    def __repr__(self):
        return "(" + str(self.state1) + "," + str(self.state2) + ")"

    def __eq__(self, other):
        if isinstance(other, ProductState):
            return self.state1 == other.state1 and self.state2 == other.state2


class LocationState(ProductState):
    def __init__(self, row, col):
        ProductState.__init__(self, row, col)

    def get_location(self):
        return self.state1, self.state2


class AgentEnvironmentProductState(ProductState):
    def __init__(self, agent_state, environment_state):
        ProductState.__init__(self, agent_state, environment_state)

    def get_agent_state(self):
        return self.state1

    def get_environment_state(self):
        return self.state2


class AgentState(LocationState):
    def __init__(self, row, col):
        LocationState.__init__(self, row, col)
        self.left_state = None
        self.right_state = None
        self.up_state = None
        self.down_state = None

    def move(self, state):
        if isinstance(state, MoveLeftState):
            return self.left_state
        elif isinstance(state, MoveRightState):
            return self.right_state
        elif isinstance(state, MoveUpState):
            return self.up_state
        elif isinstance(state, MoveDownState):
            return self.down_state
        elif isinstance(state, NotMoveState):
            return self

    def set_left_state(self, left_state):
        self.left_state = left_state

    def set_right_state(self, right_state):
        self.right_state = right_state

    def set_up_state(self, up_state):
        self.up_state = up_state

    def set_down_state(self, down_state):
        self.down_state = down_state

    def get_adjacent_states(self):
        return [self.left_state, self.right_state, self.up_state, self.down_state]

    def __repr__(self):
        return "Agent at " + LocationState.__repr__(self)


class FoodState(LocationState):
    def __init__(self, row, col):
        LocationState.__init__(self, row, col)

    def __repr__(self):
        return "Food at " + LocationState.__repr__(self)


class NoFoodState(State):
    def __init__(self):
        State.__init__(self)

    def __eq__(self, other):
        return isinstance(other, NoFoodState)

    def __repr__(self):
        return "No food"


class ActionState(State):
    pass


class AgentActionState(ActionState):
    pass


class SystemActionState(ProductState):
    def __init__(self, system_state, action_state):
        ProductState.__init__(self, system_state, action_state)

    def get_system_state(self):
        return self.state1

    def get_action_state(self):
        return self.state2


class MoveLeftState(AgentActionState):
    def __init__(self):
        AgentActionState.__init__(self)

    def __repr__(self):
        return "Agent moving left"

    def __eq__(self, other):
        return isinstance(other, MoveLeftState)


class MoveRightState(AgentActionState):
    def __init__(self):
        AgentActionState.__init__(self)

    def __repr__(self):
        return "Agent moving right"

    def __eq__(self, other):
        return isinstance(other, MoveRightState)


class MoveUpState(AgentActionState):
    def __init__(self):
        AgentActionState.__init__(self)

    def __repr__(self):
        return "Agent moving up"

    def __eq__(self, other):
        return isinstance(other, MoveUpState)


class MoveDownState(AgentActionState):
    def __init__(self):
        AgentActionState.__init__(self)

    def __repr__(self):
        return "Agent moving down"

    def __eq__(self, other):
        return isinstance(other, MoveDownState)


class NotMoveState(AgentActionState):
    def __init__(self):
        AgentActionState.__init__(self)

    def __repr__(self):
        return "Agent not moving"

    def __eq__(self, other):
        return isinstance(other, NotMoveState)


def make_agent_states(n_row, n_col):
    states = [AgentState(i, j) for i in range(n_row) for j in range(n_col)]
    for i in range(n_row):
        for j in range(n_col):
            if i > 0:
                states[i * n_col + j].set_up_state(states[(i - 1) * n_col + j])
            else:
                states[i * n_col + j].set_up_state(states[i * n_col + j])
            if i < n_row - 1:
                states[i * n_col + j].set_down_state(states[(i + 1) * n_col + j])
            else:
                states[i * n_col + j].set_down_state(states[i * n_col + j])
            if j > 0:
                states[i * n_col + j].set_left_state(states[i * n_col + (j - 1)])
            else:
                states[i * n_col + j].set_left_state(states[i * n_col + j])
            if j < n_col - 1:
                states[i * n_col + j].set_right_state(states[i * n_col + (j + 1)])
            else:
                states[i * n_col + j].set_right_state(states[i * n_col + j])
    return states


def make_system_states(agent_states, env_states):
    return [AgentEnvironmentProductState(s1, s2) for s1 in agent_states for s2 in env_states]


def make_env_states(n_row, n_col):
    return [FoodState(0, 0), FoodState(n_row-1, 0), FoodState(n_row-1, n_col-1), FoodState(0, n_col-1), NoFoodState()]


def make_agent_action_states():
    return [MoveLeftState(), MoveRightState(), MoveUpState(), MoveDownState(), NotMoveState()]


def make_system_action_product_states(system_states, action_states):
    return [SystemActionState(s1, s2) for s1 in system_states for s2 in action_states]


