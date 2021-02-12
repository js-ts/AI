import math

from .optimizer import Optimizer


class Scheduler(object):
    
    def __init__(self, optimizer, ):
        self.optimizer = optimizer

    def step(self, ):
        raise NotImplementedError
