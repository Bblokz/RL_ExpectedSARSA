import random
import numpy as np


class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha, gamma):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))
        pass

    def get_max_Q(self, state):
        # get max Q value for a given state.
        return np.max(self.Q[state, :])
        
    def select_action(self, state):
        # use epsilon greedy policy.
        if random.random() < self.epsilon:
            a = random.choice(range(self.n_actions))
        else:
            a = np.argmax(self.Q[state, :])
            indices = np.where(self.Q[state,:] == self.Q[state,a])
            if (len(indices[0]) > 1): # We have multiple options to pick
                a = np.random.choice(indices[0],1)[0] #select a random action from the options that have equal probability
        return a
        
    def update(self, state, action, reward, newState):
        # use Q learning udpate on Q table
        self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * self.get_max_Q(newState) - self.Q[state, action])


class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha, gamma):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        # use epsilon greedy policy.
        if random.random() < self.epsilon:
            a = random.choice(range(self.n_actions))
        else:
            a = np.argmax(self.Q[state, :])
            indices = np.where(self.Q[state,:] == self.Q[state,a])
            if (len(indices[0]) > 1): # We have multiple options to pick
                a = np.random.choice(indices[0],1)[0] #select a random action from the options that have equal probability
        return a
        
    def update(self, state, action, reward, newState):
        # Use SARSA to update the Q table on the last state using the reward and forecast of the reward on the new state.
        self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * self.Q[newState, self.select_action(newState)] - self.Q[state, action])


class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha, gamma):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))
        pass
        
    def select_action(self, state):
        # use epsilon greedy policy.
        if random.random() < self.epsilon:
            a = random.choice(range(self.n_actions))
        else:
            a = np.argmax(self.Q[state, :])
            indices = np.where(self.Q[state,:] == self.Q[state,a])
            if (len(indices[0]) > 1): # We have multiple options to pick
                a = np.random.choice(indices[0],1)[0] #select a random action from the options that have equal probability
        return a
    
    # Returns the probability of taking a given action in a given state.
    def getProbability(self, state, action):
        if np.argmax(self.Q[state, :]) == action:
            return 1 - self.epsilon + self.epsilon / self.n_actions
        else:
            return self.epsilon / self.n_actions
        
    def getExpectedReward(self, state):
        # loop through all actions in the state and sum the probability of taking that action and the Q table reward value
        expectedReward = 0
        for action in range(self.n_actions):
            expectedReward += self.getProbability(state, action) * self.Q[state, action]
        return expectedReward
        
    def update(self, state, action, reward, newState):
        # Use expected sarsa to update Q table using the probability of taking a given action in a given future state
        # and multiply that probabily by the Q table reward value sum over all those rewards and subtract Q(s, a).
        self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * self.getExpectedReward(newState) - self.Q[state, action])
        pass
