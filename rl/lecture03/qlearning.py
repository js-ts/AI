import random
import numpy as np 
import math
from collections import defaultdict


class QLearningAgent(object):

    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        '''
        '''
        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon 
        self.discount = discount


    def get_qvalue(self, state, action):
        '''
        '''
        return self._qvalues[state][action]
    
    def set_qvalue(self, state, action, value):
        '''
        '''
        self._qvalues[state][action] = value
    

    def get_value(self, state):
        '''
        '''
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return 0.
        return max([self.get_qvalue(state, a) for a in possible_actions])

        # SARSA
        # next_a = self.get_action(state)
        # return self.get_qvalue(state, next_a)

    
    def update(self, state, action, reward, next_state):
        '''
        '''
        gamma = self.discount
        learning_rate = self.alpha

        value = (1 - learning_rate) * self.get_qvalue(state, action) + learning_rate * (reward + gamma * self.get_value(next_state))
        # value = self.get_qvalue(state, action) + learning_rate * (reward + gamma * self.get_value(next_state) - self.get_qvalue(state, action))

        self.set_qvalue(state, action, value)


    def get_best_action(self, state):
        '''
        '''
        possible_actions = self.get_legal_actions(state)

        if len(possible_actions) == 0:
            return None
        
        qvalues = [self.get_qvalue(state, a) for a in possible_actions]

        return possible_actions[np.argmax(qvalues)]


    def get_action(self, state):
        '''
        '''
        possible_actions = self.get_legal_actions(state)

        if len(possible_actions) == 0:
            return None
        
        if len(possible_actions) == 1:
            return possible_actions[0]

        epsilon = self.epsilon
        qvalues = [self.get_qvalue(state, a) for a in possible_actions]
        i = np.argmax(qvalues)

        p = [epsilon / (len(qvalues) - 1) for _ in range(len(qvalues))]
        p[i] = 1 - epsilon

        return np.random.choice(possible_actions, p=p)


class SARSAAgent(QLearningAgent):

    def get_value(self, state):
        '''
        '''
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return 0.
        
        next_a = self.get_action(state)

        return self.get_qvalue(state, next_a)
        
    
class EVSARSAAgent(QLearningAgent):

    def softmax(self, a):
        '''
        '''
        a = np.array(a)
        a = a - np.max(a)
        exp_a = np.exp(a)
        return exp_a / sum(exp_a)

    def get_value(self, state):
        '''
        '''
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return 0.
        
        qvalues = [self.get_qvalue(state, a) for a in possible_actions]
        probs = self.softmax(qvalues)

        return (np.array(qvalues) * probs).sum()


class ReplayBuffer(object):

    def __init__(self, size):
        '''
        '''
        self._storage = []
        self._maxsize = size
    
    def __len__(self, ):
        '''
        '''
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp, done):
        '''
        '''
        data = (obs_t, action, reward, obs_tp, done)

        if len(self) == self._maxsize:
            self._storage.pop(0)
        
        self._storage.append(data)


    def sample(self, batch_size):
        '''
        '''
        if len(self) < batch_size:
            batch_size = len(self)

        index = np.random.permutation(np.arange(len(self)))[:batch_size]
        states, actions, rewards, next_states, done = map(np.array, zip(*[self._storage[i] for i in index]))

        return states, actions, rewards, next_states, done



import gym
env = gym.make('Taxi-v3')
# agent = QLearningAgent(0.5, 0.25, 0.99, lambda s: range(env.action_space.n))
agent = EVSARSAAgent(0.5, 0.25, 0.99, lambda s: range(env.action_space.n))
buffer = ReplayBuffer(1000)


def play_and_train(env, agent, replay=True, batch_size=12, t_max=10**4):

    '''
    '''
    total_reward = 0.0
    s = env.reset()

    for _ in range(t_max):
        '''
        '''
        a = agent.get_action(s)
        next_s, r, done, _ = env.step(a)
        
        agent.update(s, a, r, next_s)

        if replay:
            buffer.add(s, a, r, next_s, done)
            if batch_size > len(buffer):
                batch_size = len(buffer)
            _s, _a, _r, _next_s, _done = buffer.sample(batch_size)
            for i in range(batch_size):
                if not _done[i]:
                    agent.update(_s[i], _a[i], _r[i], _next_s[i])
        s = next_s
        total_reward += r 

        if done:
            break

    return total_reward


rewards = []
for i in range(1000):
    r = play_and_train(env, agent)
    rewards.append(r)

    agent.epsilon *= 0.99


print(np.mean(rewards[-10:]))
