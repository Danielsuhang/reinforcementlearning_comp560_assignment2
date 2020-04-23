from random import random


class MarkovDecisionProcess():
    def __init__(self, file_path):
        self.process_inputs(file_path)

    def process_inputs(self, file_path):
        f = open(file_path, "r")
        all_inputs = MarkovDecisionProcess.read_new_line(f)
        self.actions = MarkovDecisionProcess.process_action_input(all_inputs)
        self.state_names = MarkovDecisionProcess.get_state_names(all_inputs)
        self.states = MarkovDecisionProcess.init_states(self.state_names, self.actions)
        self.print_state_variables()
    
    def print_state_variables(self):
        for state in self.states:
            print (state.name)
            for action in state.possible_actions:
                print (action.cur_state, action.action, action.new_state, action.probability)
            print()

    @staticmethod
    def process_action_input(raw_inputs):
        actions = []
        for input in raw_inputs:
            unprocessed_inputs = input.split("/")
            action = Action(
                unprocessed_inputs[0], unprocessed_inputs[1], unprocessed_inputs[2], unprocessed_inputs[3])
            actions.append(action)
        return actions

    @staticmethod
    def get_state_names(raw_inputs):
        state_names = set()
        for input in raw_inputs:
            unprocessed_inputs = input.split("/")
            state_names.add(unprocessed_inputs[0])
            state_names.add(unprocessed_inputs[2])
        return state_names

    @staticmethod
    def init_states(state_names, all_actions):
        states = []
        for state_name in state_names:
            states.append(MarkovDecisionProcess.create_state(
                state_name, all_actions))
        return states

    @staticmethod
    def create_state(state_name, all_actions):
        actions = list(filter(lambda action: action.cur_state ==
                              state_name, all_actions))
        new_state = State(state_name, actions)
        return new_state

    @staticmethod
    def read_new_line(file):
        all_inputs = []
        while True:
            try:
                c_input = file.readline().rstrip().lstrip()
                if c_input.strip() != '':
                    all_inputs.append(c_input)
                else:
                    break
            except ValueError:
                print("Invalid Input")
        return all_inputs


class Action():
    def __init__(self, cur_state, action, new_state, prob):
        self.cur_state = cur_state
        self.new_state = new_state
        self.probability = prob
        self.action = action


class State():
    # Continue until all children are not -1
    # Use action score to weight probability of choosing that action
    actions = {}  # at, past, left init with -1
    name = ""

    def __init__(self, name, actions):
        self.possible_actions = actions
        self.name = name


if __name__ == "__main__":
    markov_decision_process = MarkovDecisionProcess("test_data.txt")

