import random

class MarkovDecisionProcess():

    def __init__(self, file_path, model_based=False):
        self.process_inputs(file_path)
        self.epsilon = .002
        if not model_based:
            self.total_score = 0
            self.state_name_to_obj = {state.name : state for state in self.states}
            total_runs = 2000
            exploit_explore_ratio = 0.5
            for _ in range(int(total_runs * exploit_explore_ratio)):
                for cur_state in self.states:
                    if (cur_state.name == "In"):
                        continue
                    for cur_action_name in cur_state.unique_actions:
                        self.explore_policy_model_free(cur_state, cur_action_name)
            for _ in range(int(total_runs * (1 - exploit_explore_ratio))):
                self.exploit_policy_model_free()
            self.print_state_variables()
            print()
            print(self.total_score)
        else:
            self.state_name_to_obj = {state.name : state for state in self.states}
            #for model-based, we need to learn transition probabilities
            for _ in range(3000):
                iter_diff = 0
                for cur_state in self.states:
                    iter_diff += self.model_based_learning(cur_state, _)
                if iter_diff < self.epsilon:
                    print("stopped after ", _, " iterations with diff: ", iter_diff)
                    break

                
            self.print_state_variables(True)

    def get_state_by_name(self, state_name):
        for state in self.states:
            if state.name == state_name:
                return state

    def model_based_learning(self, start_state, iteration, exploration=.2, gamma=.5):
        cur_state = start_state
        iteration_utility_diff = 0
        
        while (cur_state.name != "In"):
            start_utility = cur_state.utility
            #step 1, determine what action we will take by what is currently the most optimal or random choice if doing exploration
            next_action_name = ""
            if random.random() <= exploration:
                next_action_name = list(cur_state.unique_actions)[random.randint(0, len(cur_state.unique_actions) - 1)]
            else:
                next_action_name = self.get_optimal_action(cur_state)

            #step 2: observe what the next state will be
            next_state = self.get_next_state_from_action(cur_state, next_action_name)

            #print(cur_state.name, next_action_name, "state and action for iteration")
            #step 3: update transition probabilities
            cur_actions = list(filter(lambda a: a != None, [possible_action if possible_action.action_name == next_action_name else None for possible_action in cur_state.possible_actions_blank]))

            #for act in cur_actions:
            #    print(act.cur_state, act.new_state, act.action_name, act.probability)

            for act in cur_actions:
                act.total_observed += 1
                if self.get_state_by_name(act.new_state).name == next_state.name:
                    act.times_observed += 1
                act.update_prob()

            #step 4: update utility of current state
            best_action_utility = float("inf")
            for act in cur_state.unique_actions:
                act_utility  = 0
                cur_actions = list(filter(lambda a: a != None, [possible_action if possible_action.action_name == act else None for possible_action in cur_state.possible_actions_blank]))
                if len(cur_actions) == 0:
                    exit(-1)
                for pos_action in cur_actions:
                    act_utility += pos_action.probability * self.get_state_by_name(pos_action.new_state).utility
                if best_action_utility > act_utility: #we want low utilities
                    best_action_utility = act_utility
            cur_state.utility = best_action_utility * gamma + cur_state.DEFAULT_REWARD_VALUE

            #for act in cur_actions:
            #    print(act.cur_state, act.new_state, act.action_name, act.probability)

            #if iteration < 3:
            #    print(cur_state.utility, "post")

            end_utility = cur_state.utility
            iteration_utility_diff += abs(end_utility - start_utility)
            #step 5: go to next state
            cur_state = next_state
        return iteration_utility_diff
    
    def get_blank_action(self, start_state, action_name, next_state):
        for action in self.blank_actions:
                if action.cur_state.name == start_state.name and action_name == action.action_name and next_state.name == action.new_state:
                    return action
        print("ERROR IN GET_BLANK_ACTION")

    def get_optimal_action(self, state):
        best_action_name = ""
        best_action_val = float("inf")
        for action_name in state.unique_actions:
            val = 0
            cur_actions = list(filter(lambda a: a != None, [possible_action if possible_action.action_name == action_name else None for possible_action in state.possible_actions_blank]))
            for action in cur_actions:
                val += self.get_state_by_name(action.new_state).utility * action.probability
            if val < best_action_val:
                best_action_val = val
                best_action_name = action_name
        return best_action_name

    def exploit_policy_model_free(self):
        """Always pick the optimal action from a random state"""
        cur_state = random.choice(self.states)
        score = 0
        while (cur_state.name != "In"):
            optimal_action = self.get_optimal_action(cur_state)
            cur_state = self.get_next_state_from_action(cur_state, optimal_action)
            score += 1
        self.total_score += score
                    
    def explore_policy_model_free(self, start_state, start_action_name):
        """ Evaluate the policy of one state with one start_action_choosen
         Iteratively continue until "In" is reached """
        cur_state = start_state
        next_action_name = start_action_name
        score = 0
        while (cur_state.name != "In"):
            cur_state = self.get_next_state_from_action(
                cur_state, next_action_name)
            next_action_name = self.randomly_get_next_action_from_utilities(
                cur_state)
            score += 1
        start_state.action_utility_scores[start_action_name].append(score) 
        self.total_score += score
    
    def randomly_get_next_action_from_utilities(self, state):
        if (state.name == "In"):
            return None
        average_action_score_dict = {action_name : state.get_average_score(action_name) for action_name in state.unique_actions}
        random_num = random.uniform(
            0, sum(average_action_score_dict.values()))
        cur_cumulative_score = 0
        for action_name in state.action_utility_scores.keys():
            cur_cumulative_score += average_action_score_dict[action_name]
            if (random_num <= cur_cumulative_score):
                return action_name
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
        self.blank_actions = MarkovDecisionProcess.process_action_input(all_inputs, True)
        self.state_names = MarkovDecisionProcess.get_state_names(all_inputs)
        self.states = MarkovDecisionProcess.init_states(
            self.state_names, self.actions, self.blank_actions)

    def print_state_variables(self, model_based=False):
        total_utility = 0
        for state in self.states:
            print(state.name, " utility: ", state.utility, " optimal action from state: " + self.get_optimal_action(state))
            total_utility += state.utility
            for action in state.possible_actions if not model_based else state.possible_actions_blank:
                print(action.cur_state, action.action_name,
                      action.new_state, action.probability)
            for action in state.action_utility_scores.keys():
                if not model_based:
                    print(action, state.get_average_score(action))
            print()
        print("Total Utility: ", total_utility)

    @staticmethod
    def process_action_input(raw_inputs, blank=False):
        actions = []
        for input in raw_inputs:
            unprocessed_inputs = input.split("/")
            if not blank:
                action = Action(
                    unprocessed_inputs[0], unprocessed_inputs[1], unprocessed_inputs[2], unprocessed_inputs[3])
                actions.append(action)
            else:
                action = Action(
                    unprocessed_inputs[0], unprocessed_inputs[1], unprocessed_inputs[2], .3333)
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
    def init_states(state_names, all_actions, blank_actions):
        states = []
        for state_name in state_names:
            states.append(MarkovDecisionProcess.create_state(
                state_name, all_actions, blank_actions))
        return states

    @staticmethod
    def create_state(state_name, all_actions, blank_actions):
        actions = list(filter(lambda action: action.cur_state ==
                              state_name, all_actions))
        blank_actions = list(filter(lambda action: action.cur_state == state_name, blank_actions))
        new_state = State(state_name, actions, blank_actions)
        if new_state.name == "In":
            new_state.DEFAULT_UTILITY_SCORE = 0
            new_state.DEFAULT_REWARD_VALUE = 0
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

        self.times_observed = 0
        self.total_observed = 0

    def update_prob(self):
        self.probability = float(self.times_observed / self.total_observed)
        if self.probability == 0:
            self.probability = .33 #this is used to make sure utilities don't rush to 0,

class State():
    DEFAULT_REWARD_VALUE = 1
    DEFAULT_UTILITY_SCORE = 1
    """ 
        name: Name of this action E.g: Putt, At, Past, etc..
        possible_actions: list of Action's that the player can do at this state
        unique_actions: set of action names (string) that the player can do at this state
        action_utility_scores: Dictionary of action names to their utility score
        policy: The current best action in this state
    """
    def __init__(self, name, possible_actions, possible_actions_blank=[]):
        self.name = name
        self.possible_actions = possible_actions 
        self.possible_actions_blank = possible_actions_blank
        self.unique_actions = {
            action.action_name for action in self.possible_actions}
        self.action_utility_scores = { 
            action: [self.DEFAULT_UTILITY_SCORE] for action in self.unique_actions}
        self.policy = ""
        self.unique_action_probs = [1 / len(possible_actions) for i in range(len(possible_actions))]
        self.utility = 1 if name != "In" else 0
        self.DEFAULT_REWARD_VALUE = 1 if name != "In" else 0
    
    def get_average_score(self, action_name):
        scores = self.action_utility_scores[action_name]
        return sum(scores) / len(scores)

if __name__ == "__main__":
    markov_decision_process = MarkovDecisionProcess("test_data.txt", False)
    markov_decision_process = MarkovDecisionProcess("test_data.txt", True)
