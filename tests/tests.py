import unittest
from src.automat import RegularExpression, State, Transition, NFA, DFA, CDFA, Alphabet


class TestRegularExpression(unittest.TestCase):
    def setUp(self):
        self.regex = RegularExpression("(ab + c*.(l + d)*)*")
        self.rpn = self.regex.regex_to_rpn()

    def test_regex_to_rpn(self):
        """Test conversion of regex to Reverse Polish Notation."""
        self.assertEqual(self.rpn, "ab c * l d + * . + *")


class TestNFA(unittest.TestCase):
    def setUp(self):
        self.state0 = State(0)
        self.state1 = State(1)
        self.state2 = State(2)
        self.state3 = State(3)

        self.states = [self.state0, self.state1, self.state2, self.state3]
        self.alphabet = Alphabet(['a', 'b'])
        self.start_state = self.state0
        self.final_states = [self.state3]

        self.transitions = [
            Transition(self.state0, self.state1, 'a'),
            Transition(self.state1, self.state2, 'b'),
            Transition(self.state2, self.state3, 'a'),
            Transition(self.state0, self.state3, 'ε'),
            Transition(self.state3, self.state1, 'b')
        ]

        self.nfa = NFA(self.states, self.alphabet, self.transitions, self.start_state, self.final_states)

    def test_to_DFA(self):
        """Test conversion of NFA to DFA."""
        dfa = self.nfa.to_DFA()

        self.assertGreaterEqual(len(dfa.states), 1)
        self.assertIsNotNone(dfa.start_state)

        expected_transitions = [
            Transition(State(0), State(1), 'a'),
            Transition(State(1), State(2), 'b'),
            Transition(State(2), State(3), 'a'),
            Transition(State(0), State(1), 'b'),
        ]

        for et in expected_transitions:
            self.assertIn(et, dfa.transitions)

        for final_state in dfa.final_states:
            self.assertIn(final_state, dfa.states)

    def test_dfa_no_epsilon_transitions(self):
        """Ensure DFA has no ε-transitions."""
        dfa = self.nfa.to_DFA()
        for transition in dfa.transitions:
            self.assertNotEqual(transition.word, 'ε')

    def test_dfa_final_states(self):
        """Test that DFA has final states."""
        dfa = self.nfa.to_DFA()
        self.assertGreaterEqual(len(dfa.final_states), 1)

        for final_state in dfa.final_states:
            self.assertIn(final_state, dfa.states)


class TestCDFA(unittest.TestCase):
    def setUp(self):
        self.state0 = State(0)
        self.state1 = State(1)
        self.state2 = State(2)
        self.state3 = State(3)

        self.states = [self.state0, self.state1, self.state2, self.state3]
        self.alphabet = Alphabet(['a', 'b'])
        self.start_state = self.state0
        self.final_states = [self.state3]

        self.transitions = [
            Transition(self.state0, self.state1, 'a'),
            Transition(self.state1, self.state2, 'b'),
            Transition(self.state2, self.state3, 'a'),
            Transition(self.state0, self.state3, 'ε'),
            Transition(self.state3, self.state1, 'b')
        ]

        self.nfa = NFA(self.states, self.alphabet, self.transitions, self.start_state, self.final_states)
        self.cdfa = self.nfa.to_DFA().to_complete_dfa()

    def test_complement(self):
        """Test the complement of a CDFA."""
        complemented = self.cdfa.complement()

        self.assertNotEqual(complemented.final_states, self.cdfa.final_states)

    def test_minimize(self):
        """Test minimization of CDFA."""
        minimized = self.cdfa.minimize()

        self.assertLessEqual(len(minimized.states), len(self.cdfa.states))


    def test_to_regex(self):
        """Test conversion from CDFA to regex."""
        regex = self.cdfa.to_regex()
        self.assertIsInstance(regex, str)
        self.assertGreater(len(regex), 0)

        self.assertIn('a', regex)
        self.assertIn('b', regex)
