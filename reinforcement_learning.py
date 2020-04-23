from random import random

class MarkovDecisionProcess():
    def __init__(self, file_path):
        self.read_inputs(file_path)

    def read_inputs(self, file_path):
        f = open(file_path, "r")
        actions = self.process_action_input(MarkovDecisionProcess.read_new_line(f))
        for action in actions:
            print(action.cur_state + action.probability)
    
    def process_action_input(self, inputs):
        actions = []
        for input in inputs:
            unprocessed_actions = input.split("/")
            action = Action(unprocessed_actions[0], unprocessed_actions[1], unprocessed_actions[2], unprocessed_actions[3])
            actions.append(action)
        return actions

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
                print ("Invalid Input")
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
    actions = {} # at, past, left init with -1
    name = ""
    def __init__():
        pass





if __name__ == "__main__":
    print("Successfully ran")
    markov_decision_process = MarkovDecisionProcess("test_data.txt")



