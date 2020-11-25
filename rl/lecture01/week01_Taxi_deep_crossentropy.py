import gym
import time
import numpy as np 
import sklearn.neural_network as nn


class Agent(object):
    '''
    '''
    def __init__(self, env=None, hidden_sizes=[20, 20]):
        '''
        '''
        self.env = env
        self.hidden_sizes = hidden_sizes
        self.n_actions = env.action_space.n
        self.actions = range(env.action_space.n)
        self.policy = self.init_policy()

    def init_policy(self, ):
        '''
        '''
        policy = nn.MLPClassifier(hidden_layer_sizes=self.hidden_sizes, activation='tanh')
        policy.partial_fit([self.env.reset()] * self.n_actions, range(self.n_actions), range(self.n_actions))
        print('---init MLP done---')
        return policy

    def action(self, state):
        '''
        '''
        probs = self.policy.predict_proba(state)
        a = np.random.choice(self.actions, p=probs[0])
        
        return probs

    def action_probs(self, state):
        '''
        '''
        probs = self.policy.predict_proba(state)
        return probs     

class Solver(object):
    '''
    '''
    def __init__(self, env, agent, batches=100, epoches=10, percentile=50):
        '''
        '''
        self.env = env
        self.agent = agent
        self.batches = batches
        self.epoches = epoches
        self.percentile = percentile
        self.actions = range(self.env.action_space.n)


    def train(self, ):
        '''
        '''
        tic = time.time()
        for _ in range(self.epoches):
            states, actions, _rewards = self.generate_train_data(batches=self.batches)
            self.agent.policy.partial_fit(states, actions, self.actions)

            print(np.mean(_rewards))

            if np.mean(_rewards) > 200:
                print('---Done--Using {} s-'.format(time.time() - tic))
                break
        
        return self.agent.policy


    def generate_session(self, t_max=1000):
        '''
        '''
        states = []
        actions = []
        rewards = 0.

        s = env.reset()

        for _ in range(t_max):

            probs = self.agent.action_probs([s, ])[0]
            a = np.random.choice(self.actions, p=probs)
            s_next, r, done, _ = self.env.step(a)

            states.append(s)
            actions.append(a)
            rewards += r

            s = s_next

            if done:
                break

        return states, actions, rewards


    def select_elites(self, states, actions, rewards):
        '''
        '''
        # states, actions, rewards = map(np.array, [states, actions, rewards])
        # print(states.shape)
        # print(actions.shape)
        r_threshold = np.percentile(rewards, q=self.percentile)
        idx = (np.array(rewards) >= r_threshold).nonzero()[0]
        elite_states = [s for i in idx for s in states[i]]
        elite_actions = [a for i in idx for a in actions[i]]
        
        return np.array(elite_states), np.array(elite_actions), np.array(rewards)


    def generate_train_data(self, batches=1):
        '''
        '''
        states, actions, rewards = zip(*[self.generate_session() for _ in range(batches)])
        states, actions, _rewards = self.select_elites(states, actions, rewards)

        return states, actions, _rewards


if __name__ == '__main__':

    env = gym.make('CartPole-v0').env
    # obs0 = env.reset()
    # print(env.action_space.n)
    # print(env.observation_space.shape)
    # print(type(obs0), obs0.shape)
    
    agent = Agent(env=env)
    # probs = agent.action([obs0, env.reset()])
    # print(probs.shape)
    # print(probs.dtype)
    # print(probs[0])

    solver = Solver(env, agent, percentile=50, epoches=100)
    # sss, ass, rss = solver.generate_train_data(batches=10)
    # print(sss.shape, sss.dtype)
    # print(ass.shape, ass.dtype)
    # print(rss.shape, rss.dtype)

    solver.train()
