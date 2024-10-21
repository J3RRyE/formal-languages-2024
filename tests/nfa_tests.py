import unittest
from src.automat import *

class TestNFA(unittest.TestCase):
    def setUp(self):
        self.state0 = State(0)
        self.state1 = State(1)
        self.state2 = State(2)
        self.state3 = State(3)

        self.states = [self.state0, self.state1, self.state2, self.state3]
        self.alphabet = ['a', 'b']
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

        self.assertIn(dfa.final_states[0], dfa.states)

    def test_dfa_no_epsilon_transitions(self):
        dfa = self.nfa.to_DFA()
        for transition in dfa.transitions:
            self.assertNotEqual(transition.word, 'ε')

    def test_dfa_final_states(self):
        dfa = self.nfa.to_DFA()
        self.assertGreaterEqual(len(dfa.final_states), 1)

        for final_state in dfa.final_states:
            self.assertIn(final_state, dfa.states)

