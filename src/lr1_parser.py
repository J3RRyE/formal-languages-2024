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
        pass

    def predict(self, tokens) -> bool:
        pass

    def __first(self, start: str, token, visited: set):
        pass

    def __closure(self, situation: LR1_Situation) -> set:
        pass

    def __process_automata(self):
        pass

    def __process_table(self):
        pass

