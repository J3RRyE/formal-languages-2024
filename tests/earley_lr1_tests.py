from src.grammatics import Grammatics, Rule
from src.early_parser import Earley_Parser
from src.lr1_parser import LR1_Parser
import json

import unittest


def parse_simple_grammar_file(test_data) -> Grammatics:
    non_terminals = set(["S", "A", "B", "C", "D", "F", "E", "X", "Y", "K", "L", "M"])
    terminals = set(["a", "b", "c", "d", "(", ")"])

    mode = Grammatics.WORDS if test_data["mode"] == "WORDS" else Grammatics.SENTENCES

    rule_list = []
    if mode == Grammatics.WORDS:
        for rule_str in test_data['rules']:
            rule_list.append(Rule.parse_rule_symbols(rule_str))
    elif mode == Grammatics.SENTENCES:
        for rule_str in test_data['rules']:
            rule_list.append(Rule.parse_rule_words(rule_str))

    return Grammatics(terminals, non_terminals, "S", rule_list, mode)

class TestEarleyParser(unittest.TestCase):
    def test_Earley_algorithm(self):
        for i in range(2):
            with open('tests/General/' + str(i + 1) + '.json', 'r') as test_file:
                print('tests/General/' + str(i + 1) + '.json')

                test_data = json.load(test_file)

                grammar = parse_simple_grammar_file(test_data)
                parser = Earley_Parser().fit(grammar)

                for j in range(len(test_data['words'])):
                    ans = parser.predict(test_data['words'][j])
                    real_ans = test_data['answers'][j]
                    self.assertEqual(ans, real_ans, "word: %s, General grammar test number: %d" % (test_data['words'][j], i + 1))

        for i in range(4):
            with open('tests/Earley/' + str(i + 1) + '.json', 'r') as test_file:
                print('tests/Earley/' + str(i + 1) + '.json')

                test_data = json.load(test_file)

                grammar = parse_simple_grammar_file(test_data)
                parser = Earley_Parser().fit(grammar)

                for j in range(len(test_data['words'])):
                    ans = parser.predict(test_data['words'][j])
                    real_ans = test_data['answers'][j]
                    self.assertEqual(ans, real_ans, "word: %s, Early grammar test number: %d" % (test_data['words'][j], i + 1))

        for i in range(3):
            with open('tests/Sentences/' + str(i + 1) + '.json', 'r') as test_file:
                print('tests/Sentences/' + str(i + 1) + '.json')

                test_data = json.load(test_file)

                non_terminals = set(test_data["non_terminals"])
                terminals = set(test_data["terminals"])

                mode = Grammatics.WORDS if test_data["mode"] == "WORDS" else Grammatics.SENTENCES

                rule_list = []
                if mode == Grammatics.WORDS:
                    for rule_str in test_data['rules']:
                        rule_list.append(Rule.parse_rule_symbols(rule_str))
                elif mode == Grammatics.SENTENCES:
                    for rule_str in test_data['rules']:
                        rule_list.append(Rule.parse_rule_words(rule_str))

                grammar = Grammatics(terminals, non_terminals, test_data["start"], rule_list, mode)
                parser = Earley_Parser().fit(grammar)

                for j in range(len(test_data['words'])):
                    ans = parser.predict(test_data['words'][j].strip().split())
                    real_ans = test_data['answers'][j]
                    self.assertEqual(ans, real_ans, "word: %s, Sentences grammar test number: %d" % (test_data['words'][j], i + 1))

class TestLR1_Parser(unittest.TestCase):
    def test_LR1_algorithm(self):
        for i in range(2):
            with open('tests/General/' + str(i + 1) + '.json', 'r') as test_file:
                print('tests/General/' + str(i + 1) + '.json')

                test_data = json.load(test_file)

                grammar = parse_simple_grammar_file(test_data)
                parser = LR1_Parser().fit(grammar)

                for j in range(len(test_data['words'])):
                    ans = parser.predict(test_data['words'][j])
                    real_ans = test_data['answers'][j]
                    self.assertEqual(ans, real_ans, "word: %s, General grammar test number: %d" % (test_data['words'][j], i + 1))

        for i in range(2):
            with open('tests/LR1/' + str(i + 1) + '.json', 'r') as test_file:
                print('tests/LR1/' + str(i + 1) + '.json')

                test_data = json.load(test_file)

                grammar = parse_simple_grammar_file(test_data)
                parser = LR1_Parser().fit(grammar)

                for j in range(len(test_data['words'])):
                    ans = parser.predict(test_data['words'][j])
                    real_ans = test_data['answers'][j]
                    self.assertEqual(ans, real_ans, "word: %s, LR1 grammar test number: %d" % (test_data['words'][j], i + 1))

        for i in range(3):
            with open('tests/Sentences/' + str(i + 1) + '.json', 'r') as test_file:
                print('tests/Sentences/' + str(i + 1) + '.json')

                test_data = json.load(test_file)

                non_terminals = set(test_data["non_terminals"])
                terminals = set(test_data["terminals"])

                mode = Grammatics.WORDS if test_data["mode"] == "WORDS" else Grammatics.SENTENCES

                rule_list = []
                if mode == Grammatics.WORDS:
                    for rule_str in test_data['rules']:
                        rule_list.append(Rule.parse_rule_symbols(rule_str))
                elif mode == Grammatics.SENTENCES:
                    for rule_str in test_data['rules']:
                        rule_list.append(Rule.parse_rule_words(rule_str))

                grammar = Grammatics(terminals, non_terminals, test_data["start"], rule_list, mode)
                parser = LR1_Parser().fit(grammar)

                for j in range(len(test_data['words'])):
                    ans = parser.predict(test_data['words'][j].strip().split())
                    real_ans = test_data['answers'][j]
                    self.assertEqual(ans, real_ans, "word: %s, Sentences grammar test number: %d" % (test_data['words'][j], i + 1))
