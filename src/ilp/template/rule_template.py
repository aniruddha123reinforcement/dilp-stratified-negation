'''Defines the rule template
'''


class Rule_Template():

    def __init__(self, v: int, allow_intensional: bool, recursive: bool=False, allow_same_variables: bool=True, variables_linked: bool=False, should_have_extentional: bool=False, only_intensional: bool=False, should_include_intensional: bool=False, should_be_recursive: bool=False, same_variable: bool=False, repeated_predicates: bool=True, allowed_predicates=[], allow_negation: bool=True, base_recursion: bool=False, tail_recursion: bool=False, add_unsafe_double_negation: bool=False):
        '''

        Arguments:
            v {int} -- numberof existentially quantified variable allowed in the clause
            allow_intensional {bool} -- True is intensional predicates are allowed, False if only extensional predicates
        '''

        self._v = v
        self._allow_intensional = allow_intensional
        self._recursive = recursive
        self._allow_same_variables = allow_same_variables
        self._variables_linked = variables_linked
        self._should_have_extentional = should_have_extentional
        self._only_intensional = only_intensional
        self._should_include_intensional = should_include_intensional
        self._should_be_recursive = should_be_recursive
        self._base_recursion = base_recursion
        self._tail_recursion = tail_recursion
        self._same_variable = same_variable
        self._allowed_predicates = allowed_predicates
        self._repeated_predicates = repeated_predicates
        self._allow_negation = allow_negation
        self._add_unsafe_double_negation = add_unsafe_double_negation


    @property
    def v(self):
        return self._v

    @property
    def allow_intensional(self):
        return self._allow_intensional

    @property
    def recursive(self):
        return self._recursive

    @property
    def allow_same_variables(self):
        return self._allow_same_variables

    @property
    def variables_linked(self):
        return self._variables_linked

    @property
    def should_have_intensional(self):
        return self._should_have_intensional

    @property
    def should_have_extentional(self):
        return self._should_have_extentional

    @property
    def only_intensional(self):
        return self._only_intensional

    @property
    def should_include_intensional(self):
        return self._should_include_intensional

    @property
    def should_be_recursive(self):
        return self._should_be_recursive

    @property
    def same_variable(self):
        return self._same_variable

    @property
    def allowed_predicates(self):
        return self._allowed_predicates

    @property
    def repeated_predicates(self):
        return self._repeated_predicates

    @property
    def allow_negation(self):
        return self._allow_negation

    @property
    def base_recursion(self):
        return self._base_recursion

    @property
    def tail_recursion(self):
        return self._tail_recursion

    @property
    def add_unsafe_double_negation(self):
        return self._add_unsafe_double_negation
