from collections import deque as deque

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
        epsilon_closures = self.epsilon_closures()

        # Create new transitions without epsilon transitions
        new_transitions = []
        for transition in self.transitions:
            if transition.word != "ε":
                for state_out, states_attainable in epsilon_closures.items():
                    if transition.state_out in states_attainable:
                        new_transitions.append(Transition(state_out, transition.state_in, transition.word))

        # Update final states based on epsilon closure
        new_final_states = set(self.final_states)
        for state in self.states:
            if any(fs in epsilon_closures[state] for fs in self.final_states):
                new_final_states.add(state)

        self.transitions = new_transitions
        self.final_states = new_final_states

    def remove_unattainable_states(self) -> None:
        reachable_states = set()
        stack = [self.start_state]
        reachable_states.add(self.start_state)

        # Perform DFS or BFS to find all reachable states
        while stack:
            current_state = stack.pop()
            for transition in self.transitions:
                if transition.state_out == current_state and transition.state_in not in reachable_states:
                    reachable_states.add(transition.state_in)
                    stack.append(transition.state_in)

        # Filter out unattainable states
        self.states = [state for state in self.states if state in reachable_states]
        self.transitions = [t for t in self.transitions if t.state_out in reachable_states and t.state_in in reachable_states]
        self.final_states = [state for state in self.final_states if state in reachable_states]

    def to_DFA(self) -> DFA:
        # Initialize DFA components
        start_state_set = frozenset(self.epsilon_closures()[self.start_state])  # Start with epsilon closure of NFA's start state
        dfa_states = {start_state_set}
        dfa_transitions = []
        dfa_final_states = set()

        self.remove_epsilon_transitions()
        self.remove_unattainable_states()

        state_queue = deque([start_state_set])
        state_mapping = {start_state_set: State(0)}  # Maps NFA state sets to new DFA states
        dfa_state_counter = 1

        # BFS to explore all subsets of NFA states
        while state_queue:
            current_set = state_queue.popleft()
            current_dfa_state = state_mapping[current_set]

            # Check if any state in the set is a final state
            if any(s in self.final_states for s in current_set):
                dfa_final_states.add(current_dfa_state)

            # Process transitions for each symbol in the alphabet
            for symbol in self.alphabet:
                next_set = set()
                for nfa_state in current_set:
                    for transition in self.transitions:
                        if transition.state_out == nfa_state and transition.word == symbol:
                            next_set.update(self.epsilon_closures()[transition.state_in])

                if next_set:
                    next_set = frozenset(next_set)

                    if next_set not in dfa_states:
                        dfa_states.add(next_set)
                        state_mapping[next_set] = State(dfa_state_counter)
                        dfa_state_counter += 1
                        state_queue.append(next_set)

                    dfa_transitions.append(Transition(current_dfa_state, state_mapping[next_set], symbol))

        return DFA(
            states=list(state_mapping.values()),
            alphabet=self.alphabet,
            transitions=dfa_transitions,
            start_state=state_mapping[start_state_set],
            final_states=list(dfa_final_states)
        )

    def epsilon_closures(self):
        # Returns the epsilon closure of a given state (all states reachable via epsilon transitions)
        epsilon_closures = {state: {state} for state in self.states}

        # Compute epsilon closure for each state
        for transition in self.transitions:
            if transition.word == "ε":
                epsilon_closures[transition.state_out].add(transition.state_in)

        # Extend epsilon closures using transitive closure
        for state in self.states:
            stack = list(epsilon_closures[state])
            while stack:
                current_state = stack.pop()
                for s in epsilon_closures[current_state]:
                    if s not in epsilon_closures[state]:
                        epsilon_closures[state].add(s)
                        stack.append(s)

        return epsilon_closures

