import random


class MarkovDecisionProcess():
    def __init__(self, file_path):
        self.process_inputs(file_path)
        self.state_name_to_obj = {state.name : state for state in self.states}

        for cur_state in self.states:
            for cur_action_name in cur_state.unique_actions:
                self.evaluate_state(cur_state, cur_action_name)

    def evaluate_state(self, start_state, start_action_name):
        # Evaluate the policy of one state with one start_action_choosen
        # Iteratively continue until "In" is reached
        cur_state = start_state
        next_action_name = start_action_name
        score = 0
        while (cur_state.name != "In"):
            cur_state = self.get_next_state_from_action(
                cur_state, next_action_name)
            next_action_name = self.randomly_get_next_action_from_utilities(
                cur_state)
            score += 1
        start_state.action_utility_scores[start_action_name] = score
        print("Successfully evaluated :" , start_state.name, " with action name of: ", start_action_name)
        print("Score of: ", score)

    def randomly_get_next_action_from_utilities(self, state):
        if (state.name == "In"):
            print("Found Correct State.")
            return None
        random_num = random.uniform(
            0, sum(state.action_utility_scores.values()))
        cur_cumulative_score = 0
        for action, utility_score in state.action_utility_scores.items():
            cur_cumulative_score += utility_score
            if (random_num <= cur_cumulative_score):
                return action
        raise Exception(
            "randomly_get_next_action_from_utilities: random number exceeded cumulative utility scores.")

    def get_next_state_from_action(self, state, action_name):
        """Gets all possible actions with 'action_name' that the state can take. Uses probability to randomly determine which action to go to"""
        cur_actions = list(filter(lambda a: a != None, [possible_action if possible_action.action_name ==
                                                        action_name else None for possible_action in state.possible_actions]))
        if (len(cur_actions) == 0):
            raise Exception("State: ", state.name, " cannot perform action: ", action_name.action_name)
        cur_cumulative_score = 0 # Sum of all cur_action probabilities should be 1
        random_num = random.uniform(0, 1) 
        for action in cur_actions:
            cur_cumulative_score += action.probability
            if (random_num <= cur_cumulative_score):
                return self.state_name_to_obj[action.new_state]
        raise Exception(
            "get_next_state_from_action: random number exceeded cumulative utility scores.")

    def process_inputs(self, file_path):
        f = open(file_path, "r")
        all_inputs = MarkovDecisionProcess.read_new_line(f)
        self.actions = MarkovDecisionProcess.process_action_input(all_inputs)
        self.state_names = MarkovDecisionProcess.get_state_names(all_inputs)
        self.states = MarkovDecisionProcess.init_states(
            self.state_names, self.actions)

    def print_state_variables(self):
        for state in self.states:
            print(state.name)
            for action in state.possible_actions:
                print(action.cur_state, action.action_name,
                      action.new_state, action.probability)
            for action, score in state.action_utility_scores.items():
                print(action, score)
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
    """ 
        cur_state: The current state to perform this action
        new_state: The new state that is reached
        probabilty: The true probability reaching the new_state when running this action
        action_name: The name of the action (E.g: Putt, At, Past, etc..)
    """
    def __init__(self, cur_state, action_name, new_state, prob):
        self.cur_state = cur_state
        self.new_state = new_state
        self.probability = float(prob)
        self.action_name = action_name

class State():
    DEFAULT_UTILITY_SCORE = 1
    """ 
        name: Name of this action E.g: Putt, At, Past, etc..
        possible_actions: list of Action's that the player can do at this state
        unique_actions: set of action names (string) that the player can do at this state
        action_utility_scores: Dictionary of action names to their utility score
        policy: The current best action in this state
    """
    def __init__(self, name, possible_actions):
        self.name = name
        self.possible_actions = possible_actions 
        self.unique_actions = {
            action.action_name for action in self.possible_actions}
        self.action_utility_scores = { 
            action: self.DEFAULT_UTILITY_SCORE for action in self.unique_actions}
        self.policy = "" 


if __name__ == "__main__":
    markov_decision_process = MarkovDecisionProcess("test_data.txt")
