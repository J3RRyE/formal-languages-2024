class State:
    def __init__(self, index):
        self.id = index

    def __eq__(self, other):
        return isinstance(other, State) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"State({self.id})"


class Transition:
    def __init__(self, state_out, state_in, word):
        self.state_out = state_out
        self.state_in = state_in
        self.word = word

    def __repr__(self):
        return f"Transition({self.state_out}, {self.state_in}, '{self.word}')"

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    def __ne__(self, other):
        return self.__repr__() != other.__repr__()


class Alphabet:
    def __init__(self, letters):
        self.letters = letters


class RegularExpression:
    def __init__(self, pattern):
        self.pattern = pattern


class DFA:
    def __init__(self, states, alphabet, transitions, start_state, final_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.final_states = final_states


class NFA:
    def __init__(self, states, alphabet, transitions, start_state, final_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.final_states = final_states

    def remove_epsilon_transitions(self) -> None:
        pass

    def remove_unattainable_states(self) -> None:
        pass
    
    def to_DFA(self) -> DFA:
        return DFA()

    def epsilon_closures(self):
       pass

