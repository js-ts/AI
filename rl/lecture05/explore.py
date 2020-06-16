from abc import ABCMeta, abstractclassmethod, abstractproperty

import numpy as np 
np.set_printoptions(precision=3, suppress=True)

import enum



class BernoulliBandit(object):
    '''
    '''
    def __init__(self, n_actions=5): # K
        self._prob = np.random.random(n_actions)
    
    @property
    def action_count(self):
        return len(self._prob)
    
    def pull(self, action):
        '''
        '''
        if np.any(np.random.random() > self._prob[action]):
            return 0.
        
        return 1. 
    
    def optimal_reward(self, ):
        return np.max(self._prob)
    
    def setup(self, ):
        '''
        '''
        pass

    def reset(self, ):
        '''
        '''
        pass



class AbstractAgent(metaclass=ABCMeta):
    def ini_actions(self, n_actions):
        self._successes = np.zeros(n_actions)
        self._failures = np.zeros(n_actions)
        self._total_pulls = 0
    
    
    @abstractproperty
    def get_action(self):
        pass

    def update(self, action, reward):
        self._total_pulls += 1
        if reward == 1:
            self._successes[action] += 1
        else:
            self._failures[action] += 1
        
    @property
    def name(self):
        return self.__class__.__name__
    

class RandomAgent(AbstractAgent):
    def get_action(self, ):
        return np.random.randint(0, len(self._successes))
    
