import numpy as np


class Markov:

    def __init__(self, num_states):
        self.transitions = np.zeros((num_states, num_states), dtype=np.float64)
        self.probabilities = np.zeros((num_states, num_states), dtype=np.float64)
        self.current_state = None
        self.num_states = num_states

    def __str__(self):
        return str(self.transitions)

    def first_visit(self, state):
        self.current_state = state

    def visit(self, state):

        if self.current_state is not None:
            self.transitions[self.current_state, state] += 1

            total = 0
            for i in range(0, self.num_states):
                total += self.transitions[self.current_state, i]

            for i in range(0, self.num_states):
                if self.transitions[self.current_state, i] > 0:
                    self.probabilities[self.current_state, i] = self.transitions[self.current_state, i]/total
                else:
                    self.probabilities[self.current_state, i] = 0

            self.current_state = state
        else:
            self.current_state = state

    def get_transitions(self):
        return self.transitions

    def get_probabilities(self):
        return self.probabilities

    def get_dict(self):
        obj = dict()
        for i in range(0,self.num_states):
            for j in range(0, self.num_states):
                trans_name = str(i) + "_" + str(j)
                obj[trans_name] = round(float(self.probabilities[i][j]),4)
        obj['current_state'] = self.current_state
        return obj