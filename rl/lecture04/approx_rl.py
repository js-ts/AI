import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import random
import time

env = gym.make("CartPole-v0").env
s = env.reset()
n_actions = env.action_space.n
state_dim = env.observation_space.shape
env.close()


class Policy(nn.Module):
    '''
    '''
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_dim)

    def forward(self, x):
        '''
        '''
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = self.layer3(out)

        return out

network = Policy(np.prod(state_dim), n_actions)


def get_action(state, epsilon=0):
    '''
    '''
    if random.random() < epsilon:
        return np.random.choice(range(n_actions), p = [1./n_actions for _ in range(n_actions)]) 
    
    state = torch.tensor([state], dtype=torch.float32)
    q_values = network(state)

    return torch.argmax(q_values, dim=-1)[0].item()

def compute_td_loss(states, actions, rewards, next_states, done, gamma=0.99):
    '''
    '''
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    done = torch.tensor(done, dtype=torch.bool)

    pred_qvalues = network(states)
    mask = torch.eye(n_actions)[actions]
    pred_qvalues = torch.masked_select(pred_qvalues, mask > 0)

    predicted_next_qvalues = network(next_states)
    next_state_values, _ = predicted_next_qvalues.max(dim=-1)
    target_qvalues_for_actions = rewards + gamma * next_state_values
    target_qvalues_for_actions = torch.where(done, rewards, target_qvalues_for_actions)
    # target_qvalues_for_actions = target_qvalues_for_actions * (1 - done.to(torch.float32)) + rewards * done.to(torch.float32)

    loss = torch.pow(pred_qvalues - target_qvalues_for_actions.detach(), 2).mean()

    return loss


opt = optim.Adam(network.parameters(), lr=0.0001)
epsilon = 0.5


def generate_session(t_max=1000, epsilon=0, train=False):
    '''
    '''
    total_reward = 0
    s = env.reset()

    for _ in range(t_max):
        a = get_action(s, epsilon=epsilon)
        next_s, r, done, _ = env.step(a)

        if train:
            opt.zero_grad()
            loss = compute_td_loss([s], [a], [r], [next_s], [done])
            loss.backward()
            opt.step()
        
        total_reward += r
        s = next_s

        if done:
            break
    
    return total_reward



for i in range(1000):

    tic = time.time()
    session_rewards = [generate_session(train=True, epsilon=epsilon) for _ in range(100)]
    
    print("{}: {}, {:.2}".format(i, np.mean(session_rewards), time.time()-tic))
    
    epsilon *= 0.99

    if np.mean(session_rewards) > 300:
        print('Win')
        break

