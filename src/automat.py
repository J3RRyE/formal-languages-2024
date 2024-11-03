from audioop import reverse
from collections import deque, defaultdict
from itertools import product
import copy

class RegularExpression:
    """
    Represents a regular expression and provides methods to:
    1. Convert the regex into Reverse Polish Notation (RPN).
    2. Convert the RPN to an equivalent NFA (Non-deterministic Finite Automaton).
    """

    def __init__(self, pattern):
        """
        Initializes the regular expression with a pattern and converts it to RPN.

        :param pattern: The regular expression pattern as a string.
        """
        self.PRIORITY = {'*': 3, '.': 2, '+': 1}  # Operator precedence: * > . > +
        self.pattern = pattern
        self.rpn = self.regex_to_rpn()  # Convert to RPN during initialization.

    def regex_to_rpn(self) -> str:
        """
        Converts the regular expression into Reverse Polish Notation (RPN).
        This makes it easier to parse and convert the expression into an NFA.

        :return: A string representing the RPN form of the regular expression.
        """
        output = []  # List to store the RPN output.
        stack = []   # Stack to hold operators during conversion.

        def apply_operator():
            """Helper function to pop operators with higher or equal precedence."""
            while stack and stack[-1] not in "()" and self.PRIORITY.get(stack[-1], 0) >= self.PRIORITY.get(char, 0):
                output.append(stack.pop())

        # Iterate over the characters in the pattern after splitting by spaces and operators.
        for i, char in enumerate(self.pattern.replace(")", " ) ").replace("(", " ( ")
                                  .replace("+", " + ").replace("*", " * ").replace(".", " . ").split()):
            if char == ' ':
                continue  # Ignore spaces.
            elif char.isalpha():  # If it's a letter, add to output.
                output.append(char)
            elif char == '(':  # Push opening parenthesis to stack.
                stack.append(char)
            elif char == ')':  # Pop until matching '(' is found.
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                stack.pop()  # Remove '(' from the stack.
            else:  # For operators, apply precedence rules.
                apply_operator()
                stack.append(char)

        # Pop remaining operators from the stack.
        while stack:
            output.append(stack.pop())

        return ' '.join(output)

    def rpn_to_nfa(self):
        """
        Converts the RPN of the regular expression into an equivalent NFA.

        :return: An NFA representing the regular expression.
        """
        stack = []  # Stack to hold intermediate NFAs.

        def create_basic_nfa(symbol, number):
            """
            Creates a basic NFA for a single symbol.

            :param symbol: The input symbol for the transition.
            :return: An NFA with two states connected by the symbol.
            """
            s1, s2 = State(2 * number), State(2 * number + 1)  # Create two states.
            return NFA(
                states=[s1, s2],
                alphabet=Alphabet([symbol]),
                transitions=[Transition(s1, s2, symbol)],
                start_state=s1,
                final_states=[s2]
            )

        # Iterate over each token in the RPN expression.
        for i, char in enumerate(self.rpn.split()):
            if char.isalpha():  # Create a basic NFA for alphabet symbols.
                stack.append(create_basic_nfa(char, i))
            elif char == '.':  # Concatenation operation.
                nfa2 = stack.pop()
                nfa1 = stack.pop()
                # Extend transitions of nfa1 with those of nfa2.
                nfa1.transitions.extend(nfa2.transitions)
                nfa1.transitions.extend([Transition(nfa1.final_states[i], nfa2.start_state, 'ε')
                                                                    for i in range(len(nfa1.final_states))])
                nfa1.alphabet.letters.extend(nfa2.alphabet.letters)
                nfa1.states.extend(nfa2.states)
                nfa1.final_states = nfa2.final_states  # Update final states.
                stack.append(nfa1)
            elif char == '+':  # Union operation.
                nfa2 = stack.pop()
                nfa1 = stack.pop()
                new_start = State(2 * i)  # New start state.
                new_final = State(2 * i + 1)  # New final state.
                # Create epsilon transitions for the union.
                transitions = [
                    Transition(new_start, nfa1.start_state, "ε"),
                    Transition(new_start, nfa2.start_state, "ε"),
                    Transition(nfa1.final_states[0], new_final, "ε"),
                    Transition(nfa2.final_states[0], new_final, "ε"),
                ]
                # Combine states and transitions into a new NFA.
                stack.append(NFA(
                    states=[new_start, new_final] + nfa1.states + nfa2.states,
                    alphabet=Alphabet(list(set(nfa1.alphabet.letters + nfa2.alphabet.letters))),
                    transitions=transitions + nfa1.transitions + nfa2.transitions,
                    start_state=new_start,
                    final_states=[new_final]
                ))
            elif char == '*':  # Kleene star operation.
                nfa = stack.pop()
                new_pre_final = State(2 * i)  # New pre final state.
                new_final = State(2 * i + 1)  # New final state.
                # Create epsilon transitions for the Kleene star.
                transitions = [
                    Transition(nfa.start_state, new_final, "ε"),
                    Transition(new_pre_final, new_final, "ε"),
                    Transition(new_pre_final, nfa.start_state, "ε")
                ]

                transitions.extend(
                    [Transition(i, new_pre_final, "ε") for i in nfa.final_states]
                )
                # Build the new NFA with added transitions.
                stack.append(NFA(
                    states=[new_pre_final, new_final] + nfa.states,
                    alphabet=nfa.alphabet,
                    transitions=transitions + nfa.transitions,
                    start_state=nfa.start_state,
                    final_states=[new_final]
                ))

        # The final NFA is the only element left on the stack.
        return stack.pop()

    def __repr__(self):
        """
        String representation of the regular expression.

        :return: The original regex pattern enclosed in parentheses.
        """
        return f"({self.pattern})"

class State:
    """
    Represents a state in an automaton.

    Attributes:
        id (int): A unique identifier for the state.
    """

    def __init__(self, index):
        """
        Initializes the state with a unique ID.

        :param index: The unique identifier for this state.
        """
        self.id = index

    def __eq__(self, other):
        """
        Checks equality between two State objects based on their IDs.

        :param other: The other state to compare with.
        :return: True if the states have the same ID, otherwise False.
        """
        return isinstance(other, State) and self.id == other.id

    def __gt__(self, other):
        """
        Compares if this State object's ID is greater than another State's ID.

        :param other: The other state to compare with.
        :return: True if this state's ID is greater than the other state's ID, otherwise False.
        """
        return self.id > other.id

    def __ge__(self, other):
        """
        Compares if this State object's ID is greater than or equal to another State's ID.

        :param other: The other state to compare with.
        :return: True if this state's ID is greater than or equal to the other state's ID, otherwise False.
        """
        return self.id >= other.id

    def __lt__(self, other):
        """
        Compares if this State object's ID is less than another State's ID.

        :param other: The other state to compare with.
        :return: True if this state's ID is less than the other state's ID, otherwise False.
        """
        return self.id < other.id

    def __le__(self, other):
        """
        Compares if this State object's ID is less than or equal to another State's ID.

        :param other: The other state to compare with.
        :return: True if this state's ID is less than or equal to the other state's ID, otherwise False.
        """
        return self.id <= other.id

    def __hash__(self):
        """
        Computes the hash value for the state based on its ID.

        :return: The hash value of the state.
        """
        return hash(self.id)

    def __repr__(self):
        """
        Returns a string representation of the state.

        :return: A string in the format 'State(id)'.
        """
        return f"State({self.id})"


class Transition:
    """
    Represents a transition in an automaton.

    Attributes:
        state_out (State): The state from which the transition originates.
        state_in (State): The state to which the transition leads.
        word (str): The symbol that triggers the transition.
    """

    def __init__(self, state_out, state_in, word):
        """
        Initializes the transition with two states and a transition symbol.

        :param state_out: The origin state of the transition.
        :param state_in: The destination state of the transition.
        :param word: The symbol causing the transition.
        """
        self.state_out = state_out
        self.state_in = state_in
        self.word = word

    def __repr__(self):
        """
        Returns a string representation of the transition.

        :return: A string in the format 'Transition(State1, State2, 'symbol')'.
        """
        return f"Transition({self.state_out}, {self.state_in}, '{self.word}')"

    def __eq__(self, other):
        """
        Checks equality between two Transition objects based on their attributes.

        :param other: The other transition to compare with.
        :return: True if both transitions are identical, otherwise False.
        """
        return self.__repr__() == other.__repr__()

    def __ne__(self, other):
        """
        Checks inequality between two Transition objects.

        :param other: The other transition to compare with.
        :return: True if transitions are different, otherwise False.
        """
        return self.__repr__() != other.__repr__()


    def __hash__(self):
        """
        Computes a hash value for the transition using the states and a unique formula.

        :return: The hash value of the transition.
        """
        return hash(self.state_out.id) * 17 + hash(self.state_in) ** 5 % 10000009

    def __iter__(self):
        """
        Provides an iterator to unpack the transition into its components.

        :return: An iterator over (state_out, state_in, word).
        """
        return iter((self.state_out, self.state_in, self.word))


class Alphabet:
    """
    Represents the set of symbols (alphabet) used in an automaton.

    Attributes:
        letters (list): A list of symbols in the alphabet.
    """

    def __init__(self, letters):
        """
        Initializes the alphabet with a set of letters.

        :param letters: A list of symbols.
        """
        self.letters = letters

    def __iter__(self):
        """
        Provides an iterator over the letters in the alphabet.

        :return: An iterator over the symbols.
        """
        return iter(self.letters)

    def __add__(self, other):
        return Alphabet(self.letters + other.letters)

    def __repr__(self):
        """
        Returns a string representation of the alphabet.

        :return: A string in the format 'Alphabet([symbols])'.
        """
        return f"Alphabet({self.letters})"


class DFA:
    """
    Represents a Deterministic Finite Automaton (DFA).

    Attributes:
        states (list): A list of State objects representing the DFA's states.
        alphabet (Alphabet): The set of symbols used in the DFA.
        transitions (list): A list of Transition objects representing state transitions.
        start_state (State): The initial state of the DFA.
        final_states (list): A list of states considered as final/accepting states.
    """

    def __init__(self, states, alphabet, transitions, start_state, final_states):
        """
        Initializes the DFA with its components.

        :param states: A list of State objects.
        :param alphabet: An Alphabet object representing the symbols.
        :param transitions: A list of Transition objects.
        :param start_state: The initial state of the DFA.
        :param final_states: A list of final/accepting states.
        """
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.final_states = final_states

    def to_complete_dfa(self):
        """
        Converts the DFA to a complete DFA by adding a trap state to handle missing transitions.

        :return: A complete DFA (CDFA) object with all possible transitions covered.
        """
        # Create a trap state that catches missing transitions.
        trap_state = State(len(self.states))

        # List to store new transitions including those to the trap state.
        new_transitions = list(self.transitions)
        self.states.append(trap_state)

        # Add transitions to the trap state for missing transitions.
        for state in self.states:
            for letter in self.alphabet.letters:
                if not any(t.state_out == state and t.word == letter for t in self.transitions):
                    new_transitions.append(Transition(state, trap_state, letter))

        # Clean isolated vertices
        count = {state: [0, 0] for state in self.states}
        for (state_out, state_in, word) in new_transitions:
            count[state_in][0] += 1
            count[state_out][1] += 1

        for (state, number) in count.items():
            if number[0] + number[1] == 0:
                for i in range(len(new_transitions)):
                    if (new_transitions[i].state_out == state) | (new_transitions[i].state_in == state):
                        new_transitions[i] = ''
                self.states.remove(state)

        new_transitions = list(set(new_transitions))
        try:
            new_transitions.remove('')
        except:
            pass

        # Return the complete DFA with the trap state and all transitions.
        return CDFA(
            states=self.states,
            alphabet=self.alphabet,
            transitions=new_transitions,
            start_state=self.start_state,
            final_states=self.final_states
        )
class NFA:
    """
    Represents a Non-deterministic Finite Automaton (NFA).

    Attributes:
        states (list): A list of State objects representing the NFA's states.
        alphabet (Alphabet): The set of symbols used in the NFA.
        transitions (list): A list of Transition objects representing state transitions.
        start_state (State): The initial state of the NFA.
        final_states (list): A list of states considered as final/accepting states.
    """

    def __init__(self, states, alphabet, transitions, start_state, final_states):
        """
        Initializes the NFA with its components.

        :param states: A list of State objects.
        :param alphabet: An Alphabet object representing the symbols.
        :param transitions: A list of Transition objects.
        :param start_state: The initial state of the NFA.
        :param final_states: A list of final/accepting states.
        """
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.final_states = final_states

    def remove_epsilon_transitions(self) -> None:
        """
        Removes epsilon transitions from the NFA by computing the epsilon closures
        and adjusting the transitions and final states accordingly.
        """
        epsilon_closures = self.epsilon_closures()

        new_transitions = []
        # Iterate through existing transitions to create new transitions
        for transition in self.transitions:
            if transition.word != "ε":  # Ignore epsilon transitions
                for state_out, states_attainable in epsilon_closures.items():
                    if transition.state_out in states_attainable:
                        # Add transition considering epsilon closures
                        new_transitions.append(Transition(state_out, transition.state_in, transition.word))

        # Determine new final states based on epsilon closures
        new_final_states = set(self.final_states)
        for state in self.states:
            if any(fs in epsilon_closures[state] for fs in self.final_states):
                new_final_states.add(state)

        # Update transitions and final states
        self.transitions = new_transitions
        self.final_states = new_final_states

    def remove_unattainable_states(self) -> None:
        """
        Removes states that cannot be reached from the start state.

        This ensures that only reachable states and transitions are retained in the NFA.
        """
        reachable_states = set()
        stack = [self.start_state]
        reachable_states.add(self.start_state)

        # Depth-first search to find all reachable states
        while stack:
            current_state = stack.pop()
            for transition in self.transitions:
                if transition.state_out == current_state and transition.state_in not in reachable_states:
                    reachable_states.add(transition.state_in)
                    stack.append(transition.state_in)

        # Update states, transitions, and final states to only include reachable states
        self.states = [state for state in self.states if state in reachable_states]
        self.transitions = [t for t in self.transitions if t.state_out in reachable_states and t.state_in in reachable_states]
        self.final_states = [state for state in self.final_states if state in reachable_states]

    def to_DFA(self) -> DFA:
        """
        Converts the NFA to a Deterministic Finite Automaton (DFA).

        :return: A DFA object representing the equivalent DFA.
        """
        # Start with the epsilon closure of the initial state
        start_state_set = frozenset(self.epsilon_closures()[self.start_state])
        dfa_states = {start_state_set}
        dfa_transitions = []
        dfa_final_states = set()

        # Remove epsilon transitions and unattainable states before conversion
        self.remove_epsilon_transitions()
        self.remove_unattainable_states()

        state_queue = deque([start_state_set])
        state_mapping = {start_state_set: State(0)}  # Maps NFA state sets to DFA states
        dfa_state_counter = 1

        # Build the DFA by processing each set of NFA states
        while state_queue:
            current_set = state_queue.popleft()
            current_dfa_state = state_mapping[current_set]

            # Check if the current set of NFA states contains a final state
            if any(s in self.final_states for s in current_set):
                dfa_final_states.add(current_dfa_state)

            # Process transitions for each symbol in the alphabet
            for symbol in self.alphabet:
                next_set = set()
                for nfa_state in current_set:
                    for transition in self.transitions:
                        if transition.state_out == nfa_state and transition.word == symbol:
                            next_set.update(self.epsilon_closures()[transition.state_in])  # Add states from epsilon closure

                if next_set:
                    next_set = frozenset(next_set)

                    if next_set not in dfa_states:
                        dfa_states.add(next_set)
                        state_mapping[next_set] = State(dfa_state_counter)
                        dfa_state_counter += 1
                        state_queue.append(next_set)

                    # Create a transition for the DFA
                    dfa_transitions.append(Transition(current_dfa_state, state_mapping[next_set], symbol))

        # Return the constructed DFA
        return DFA(
            states=list(state_mapping.values()),
            alphabet=self.alphabet,
            transitions=dfa_transitions,
            start_state=state_mapping[start_state_set],
            final_states=list(dfa_final_states)
        )

    def epsilon_closures(self):
        """
        Computes the epsilon closures for all states in the NFA.

        :return: A dictionary mapping each state to its epsilon closure set.
        """
        # Initialize epsilon closures
        epsilon_closures = {state: {state} for state in self.states}

        # Populate epsilon closures with epsilon transitions
        for transition in self.transitions:
            if transition.word == "ε":
                epsilon_closures[transition.state_out].add(transition.state_in)

        # Compute the complete epsilon closure for each state
        for state in self.states:
            stack = list(epsilon_closures[state])
            while stack:
                current_state = stack.pop()
                for s in epsilon_closures[current_state]:
                    if s not in epsilon_closures[state]:
                        epsilon_closures[state].add(s)
                        stack.append(s)

        return epsilon_closures

class CDFA:
    """
    Represents a Complete Deterministic Finite Automaton (CDFA).

    Attributes:
        states (list): A list of State objects representing the CDFA's states.
        alphabet (Alphabet): The set of symbols used in the CDFA.
        transitions (list): A list of Transition objects representing state transitions.
        start_state (State): The initial state of the CDFA.
        final_states (list): A list of states considered as final/accepting states.
    """

    def __init__(self, states, alphabet, transitions, start_state, final_states):
        """
        Initializes the CDFA with its components.

        :param states: A list of State objects.
        :param alphabet: An Alphabet object representing the symbols.
        :param transitions: A list of Transition objects.
        :param start_state: The initial state of the CDFA.
        :param final_states: A list of final/accepting states.
        """
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.final_states = final_states

    def minimize(self):
        """
        Minimizes the CDFA using the partition refinement algorithm.

        :return: A new CDFA that is the minimized version of this CDFA.
        """
        states_id = {state: i for i, state in enumerate(self.states)}
        for i in range(len(self.transitions)):
            self.transitions[i].state_out = State(states_id[self.transitions[i].state_out])
            self.transitions[i].state_in = State(states_id[self.transitions[i].state_in])

        for i in range(len(self.states)):
            self.states[i] = State(states_id[self.states[i]])

        for i in range(len(self.final_states)):
            self.final_states[i] = State(states_id[self.final_states[i]])

        self.start_state = State(states_id[self.start_state])
        is_final_state = [True if state in self.final_states else False for state in sorted(self.states)]

        def build_table(n, is_final_states, reverse_transitionss):
            queue = deque()
            marked_tmp = [[False] * n] * n
            for i in range(n):
                for j in range(n):
                    if not marked_tmp[i][j] and is_final_states[i] != is_final_states[j]:
                        marked_tmp[i][j] = marked_tmp[j][i] = True
                        queue.append((i, j))

            while queue:
                u, v = queue.popleft()
                for c in self.alphabet:
                    for r in reverse_transitionss[u][c]:
                        for s in reverse_transitionss[v][c]:
                            if not marked_tmp[r][s]:
                                marked_tmp[r][s] = marked_tmp[s][r] = True
                                queue.append((r, s))
            return marked_tmp

        reverse_transitions = { state.id: { letter : set() for letter in self.alphabet} for state in self.states }

        for (state_in, state_out, letter) in self.transitions:
            reverse_transitions[state_in.id][letter].add(state_out.id)

        marked = build_table(len(self.states), is_final_state, reverse_transitions)
        component = [-1] * len(self.states)

        for i in range(len(self.states)):
            if not marked[0][i]:
                component[i] = 0

        components_count = 0
        for i in range(len(self.states)):
            if component[i] == -1:
                components_count += 1
                component[i] = components_count
                for j in range(i + 1, len(self.states)):
                    if not marked[i][j]:
                        component[j] = components_count

        new_states = list(set([State(i) for i in component]))
        new_start_state = State(component[self.start_state.id])
        new_final_states = list(set([State(component[states_id[i]]) for i in self.final_states]))
        new_transitions = list(set([Transition(State(component[states_id[i]]),
                                               State(component[states_id[j]]),
                                               letter)
                                                for (i, j, letter) in self.transitions]))

        return CDFA(states=new_states,
                    alphabet=self.alphabet,
                    transitions=new_transitions,
                    start_state=new_start_state,
                    final_states=new_final_states)

    def complement(self):
        """
        Computes the complement of the CDFA.

        :return: A new CDFA that accepts the complement language of this CDFA.
        """
        new_final_states = [state for state in self.states if state not in self.final_states]

        return CDFA(
            states=copy.deepcopy(self.states),
            alphabet=copy.deepcopy(self.alphabet),
            transitions=copy.deepcopy(self.transitions),
            start_state=copy.deepcopy(self.start_state),
            final_states=copy.deepcopy(new_final_states)
        )

    def to_regex(self) -> str:
        """
        Converts the CDFA into an equivalent regular expression using state elimination.

        :return: A string representing the regular expression equivalent to the CDFA.
        """

        if len(self.states) == 1:
            # Collect all transition labels for the self-loops on the single state.
            labels = [t.word for t in self.transitions if
                      t.state_in == self.states[0] and t.state_out == self.states[0]]
            # Join labels with '+' for union and apply Kleene star.
            return f"({' + '.join(labels)})*"

        # Step 1: Initialize a table to hold regex expressions between states.
        n = len(self.states)
        R = [[None] * n for _ in range(n)]  # R[i][j] holds the regex from state i to state j.

        # Step 2: Populate the table with initial transitions (direct transitions or epsilon).
        for i, state in enumerate(self.states):
            for j, next_state in enumerate(self.states):
                # Collect all transitions from state `i` to state `j`.
                labels = [t.word for t in self.transitions if t.state_out == state and t.state_in == next_state]
                if labels:
                    R[i][j] = '+'.join(labels)  # Join multiple transition symbols with `+`.
                elif i == j:
                    R[i][j] = 'ε'  # Self-loop as epsilon if no other transition exists.
                else:
                    R[i][j] = ''  # No direct transition between these states.

        # Step 3: Perform state elimination.
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    # Update R[i][j] to include paths that go through state `k`.
                    direct = R[i][j]  # Direct path from i to j.
                    via_k = f"{R[i][k]}({R[k][k]})*{R[k][j]}"  # Path through k.

                    # Use union to combine paths.
                    if direct:
                        R[i][j] = f"({direct})+({via_k})" if via_k else direct
                    else:
                        R[i][j] = via_k

        # Step 4: The final regular expression is the path from the start state to any final state.
        start_index = self.states.index(self.start_state)
        final_indices = [self.states.index(f) for f in self.final_states]

        # Build the regex by combining paths to all final states.
        regex = '+'.join(R[start_index][f] for f in final_indices)

        return regex.replace('ε', '')  # Optional: Remove ε if not needed for readability.

    def _cleanup_regex(self, regex):
        """
        Cleans up the regular expression by removing unnecessary symbols.

        :param regex: The regular expression string to be cleaned.
        :return: A cleaned regular expression string.
        """
        regex = regex.replace('+ ε', '').replace('ε +', '').replace('ε', '')

        while '()' in regex:
            regex = regex.replace('()', '')

        # Remove redundant parentheses
        if regex.startswith('(') and regex.endswith(')'):
            count = 0
            balanced = True
            for char in regex[1:-1]:
                if char == '(':
                    count += 1
                elif char == ')':
                    count -= 1
                if count < 0:
                    balanced = False
                    break
            if balanced and count == 0:
                regex = regex[1:-1]

        return regex
