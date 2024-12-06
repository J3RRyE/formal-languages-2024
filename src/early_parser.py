from src.grammatics import Rule, Grammatics

class Early_Situation:
    '''
    EarlySituation - (left -> right, pointer, link).
    ptr_offset means the position of read index in the self.right string
    '''
    def __init__(self, rule: Rule, pointer_offset = 0, link = 0) -> None:
        self.rule = rule
        self.ptr_offset = pointer_offset
        self.link = link

    def __eq__(self, __value) -> bool:
        return self.rule == __value.rule and \
                        self.ptr_offset == __value.ptr_offset and \
                                self.link == __value.link

    def __repr__(self) -> str:
        return f"{self.rule.left} -> {str(self.rule.right[:self.ptr_offset])} * {str(self.rule.right[self.ptr_offset:])}"

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def completed(self) -> bool:
        return self.ptr_offset == len(self.rule.right)

    def get_cur_token(self) -> str:
        return self.rule.right[self.ptr_offset]
    
    def move_offset(self):
        self.ptr_offset = min(self.ptr_offset + 1, len(self.rule.right))

    def copy(self, mv_offset=0):
        return Early_Situation(self.rule, self.ptr_offset + mv_offset, self.link)

class Earley_Parser:
    def __init__(self):
        pass

    def fit(self, grammar: Grammatics):
        pass

    def predict(self, tokens) -> bool:
        pass

    def __scan_situations(self, index: int, token: str):
        pass

    def __predict_situations(self, index: int, situations: set[Early_Situation]) -> set[Early_Situation]:
        pass

    def __complete_situations(self, index: int, situations: set[Early_Situation]) -> set[Early_Situation]:
        pass

