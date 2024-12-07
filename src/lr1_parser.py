from src.grammatics import Rule, Grammatics

class LR1_Situation:
    '''
    LR1Situation - (left -> right, pointer_offset, precheck)
    '''
    def __init__(self, rule: Rule, pointer_offset = 0, precheck = {}):
        self.rule = rule                    # Rule (id: left -> right)
        self.ptr_offset = pointer_offset    # int
        self.precheck = precheck            # set[str]

    def __eq__(self, __value) -> bool:
        return self.rule == __value.rule and \
                        self.ptr_offset == __value.ptr_offset and \
                                self.precheck == __value.precheck

    def __repr__(self) -> str:
        return f"({self.rule.left} -> {str(self.rule.right[:self.ptr_offset])}*{str(self.rule.right[self.ptr_offset:])}, {str(self.precheck)})"

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def completed(self) -> bool:
        return self.ptr_offset >= len(self.rule.right)
    
    def is_last(self) -> bool:
        return self.ptr_offset >= max(len(self.rule.right) - 1, 0)

    def get_cur_token(self) -> str:
        if self.ptr_offset >= len(self.rule.right):
            return ""
        return self.rule.right[self.ptr_offset]

    def get_next_token(self) -> str:
        if self.ptr_offset + 1 >= len(self.rule.right):
            return ""
        return self.rule.right[self.ptr_offset + 1]

    def move_offset(self):
        self.ptr_offset = min(self.ptr_offset + 1, len(self.rule.right))

    def copy(self, mv_offset=0):
        return LR1_Situation(self.rule, min(self.ptr_offset + mv_offset, len(self.rule.right)), self.precheck)

class LR1_Vertex:
    def __init__(self, situation_set: set):
        self.situations = situation_set
        self.step = dict()

    def find_completed(self) -> set:
        ans = set()
        for situation in self.situations:
            if situation.completed():
                ans.add(situation)
        return ans

    def __repr__(self) -> str:
        return f"({str(self.situations)} : {str(self.step)})"

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __eq__(self, other) -> bool:
        return self.situations == other.situations

class LR1_Table_Vertex:
    SHIFT = 0
    REDUCE = 1
    def __init__(self, tp: int, val: int):
        self.type = tp
        self.value = val

    def __eq__(self, other) -> bool:
        return self.type == other.type and self.value == other.value

    def __repr__(self) -> str:
        string = "shift" if self.type == 0 else "reduce" if self.type == 1 else "goto" 
        string += f"({self.value})"
        return string

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return hash(self.__repr__())

class LR1_Parser:
    def __init__(self):
        pass

    def fit(self, grammar: Grammatics):
        grammar.check_grammar()
        self.grammar = grammar

        self.start = "&"
        self.grammar.non_terminals.add(self.start)

        if self.grammar.mode == Grammatics.WORDS:
            self.start_rule = Rule(self.start, self.grammar.start_token, len(self.grammar.rules))
        elif self.grammar.mode == Grammatics.SENTENCES:
            self.start_rule = Rule(self.start, self.grammar.start_token.split(), len(self.grammar.rules))

        self.grammar.rules.add(self.start_rule)
        self.rules = {non_term : {Rule(rule.left, rule.right, rule.id)\
                                    for rule in self.grammar.rules if rule.left.strip() == non_term}\
                                                    for non_term in self.grammar.non_terminals}
        self.rules.get(self.start, set()).add(self.start_rule)

        self.first = {non_term : set() for non_term in self.grammar.non_terminals}
        self.first.update({term : set() for term in self.grammar.terminals})
        for non_term in self.grammar.non_terminals:
            visited = set()
            self.__first(non_term, non_term, visited)

        self.vertices_array = []
        start_situation = LR1_Situation(self.start_rule, 0, {"$"})
        self.start_vertex = LR1_Vertex(self.__closure(start_situation))
        self.vertices_array.append(self.start_vertex)

        self.process_queue = []
        self.process_queue.append(0)
        while len(self.process_queue) > 0:
            self.__process_automata()
            self.process_queue.pop(0)

        self.table = [{} for _ in range(len(self.vertices_array))]

        self.__process_table()

        return self

    def predict(self, tokens) -> bool:
        for token in tokens:
            if token not in self.grammar.terminals:
                return False

        tokens += "$"

        stack = list()
        stack.append(0)

        for token in tokens:
            flag = True
            while flag:
                check_vertex_index = stack[-1]
                if token not in self.table[check_vertex_index].keys():
                    return False

                table_vertex = self.table[check_vertex_index][token]
                if table_vertex == LR1_Table_Vertex(LR1_Table_Vertex.REDUCE, self.start_rule.id) and token == "$":
                    return True

                if table_vertex.type == LR1_Table_Vertex.SHIFT:
                    stack.append(token) # append token
                    stack.append(table_vertex.value) # append vertex index
                    flag = False
                else:
                    rule_id = table_vertex.value
                    rule_by_id = self.grammar.get_rule(rule_id)

                    to_reduce = list(rule_by_id.right)

                    if len(stack) <= 2 * len(to_reduce):
                        return False
                    if len(to_reduce) > 0:
                        stack_checker = []

                        for token_index in range(-len(to_reduce) * 2, 0, 2):
                            stack_checker.append(stack[token_index])

                        if stack_checker != to_reduce:
                            return False
                        else:
                            stack = stack[:(-len(to_reduce) * 2)]

                    vert_index = stack[-1]
                    if rule_by_id.left not in self.table[vert_index]:
                        return False

                    stack.append(rule_by_id.left) # append token
                    stack.append(self.table[vert_index][rule_by_id.left].value) #append vertex index

        return False

    def __first(self, start: str, token, visited: set):
        if token in self.grammar.terminals:
            self.first[start].add(token)
            return

        if token in self.grammar.non_terminals:
            for rule in self.rules[token]:
                if rule in visited:
                    return

                visited.add(rule)

                if len(rule.right) > 0:
                    self.__first(start, rule.right[0], visited)

    def __closure(self, situation: LR1_Situation) -> set:
        DONE = 1
        ONWORK = 0
        situations = [set() for _ in range(2)]
        situations[ONWORK].add(situation)

        while len(situations[ONWORK]) > 0:
            current = situations[ONWORK].pop()

            if current in situations[DONE]:
                continue

            if current.completed() or current.get_cur_token() in self.grammar.terminals:
                situations[DONE].add(current)
                continue

            for rule in self.rules[current.get_cur_token()]:
                if current.is_last():
                    situations[ONWORK].add(LR1_Situation(rule, precheck=current.precheck))
                else:
                    if current.get_next_token() in self.grammar.non_terminals:
                        situations[ONWORK].add(LR1_Situation(rule, precheck=self.first[current.get_next_token()]))
                    else:
                        situations[ONWORK].add(LR1_Situation(rule, precheck={current.get_next_token()}))

            situations[DONE].add(current)

        return situations[DONE]

    def __process_automata(self):
        check_vertex_index = self.process_queue[0]
        check_vertex = self.vertices_array[check_vertex_index]

        for terminal in self.grammar.terminals:
            situation_set = set()
            for situation in check_vertex.situations:
                if situation.get_cur_token() == terminal:
                    situation_set.update(self.__closure(situation.copy(mv_offset=1)))
            if len(situation_set) > 0:
                to_vertex = LR1_Vertex(situation_set)
                if to_vertex in self.vertices_array:
                    self.vertices_array[check_vertex_index].step[terminal] = self.vertices_array.index(to_vertex)
                else:
                    to_vertex_index = len(self.vertices_array)
                    self.vertices_array.append(to_vertex)
                    self.vertices_array[check_vertex_index].step[terminal] = to_vertex_index
                    self.process_queue.append(to_vertex_index)

        for non_terminal in self.grammar.non_terminals:
            situation_set = set()
            for situation in check_vertex.situations:
                if situation.get_cur_token() == non_terminal:
                    situation_set.update(self.__closure(situation.copy(mv_offset=1)))
            if len(situation_set) > 0:
                to_vertex = LR1_Vertex(situation_set)
                if to_vertex in self.vertices_array:
                    self.vertices_array[check_vertex_index].step[non_terminal] = self.vertices_array.index(to_vertex)
                else:
                    to_vertex_index = len(self.vertices_array)
                    self.vertices_array.append(to_vertex)
                    self.vertices_array[check_vertex_index].step[non_terminal] = to_vertex_index
                    self.process_queue.append(to_vertex_index)

    def __process_table(self):
        for check_vertex_index in range(len(self.vertices_array)):
            for token, to_index in self.vertices_array[check_vertex_index].step.items():
                self.table[check_vertex_index][token] = LR1_Table_Vertex(LR1_Table_Vertex.SHIFT, to_index)
            
        for check_vertex_index in range(len(self.vertices_array)):
            for completed_situation in self.vertices_array[check_vertex_index].find_completed():
                for token in completed_situation.precheck:
                    default_table_vert = LR1_Table_Vertex(LR1_Table_Vertex.REDUCE, completed_situation.rule.id)
                    if self.table[check_vertex_index].get(token, default_table_vert) != default_table_vert:
                        if self.table[check_vertex_index].get(token, default_table_vert).type == LR1_Table_Vertex.REDUCE:
                            raise Exception("Грамматика не LR(1).")
                        continue
                    self.table[check_vertex_index][token] = default_table_vert

if __name__ == '__main__':
    N, Sigma, P = map(int, input().split())
    non_terminals = [char for char in input()]
    terminals = [char for char in input()]

    rules = [Rule.parse_rule_symbols(input()) for i in range(P)]

    start_token = input()

    grammar = Grammatics(terminals, non_terminals, start_token, rules, 0)

    parser = LR1_Parser()
    parser.fit(grammar)

    tokens_number = int(input())
    for i in range(tokens_number):
        tokens = input().strip()
        print("Yes" if parser.predict(tokens) else "No")
