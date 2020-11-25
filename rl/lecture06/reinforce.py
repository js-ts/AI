import gym
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np 



env = gym.make("CartPole-v0").env
s = env.reset()

model = nn.Sequential(
    nn.Linear(env.observation_space.shape[0], 64),
    nn.ReLU(),
    nn.Linear(64, env.action_space.n),
    # nn.ReLU()
    # nn.Softmax(dim=-1)
)

def predict_probs(states):
    '''
    '''
    states = torch.tensor(states, dtype=torch.float32)
    logits = model(states)
    preds = torch.softmax(logits, dim=-1)

    return preds.cpu().data.numpy()


def generate_session(t_max=1000):
    '''
    '''
    states = []
    actions = []
    rewars = []

    s = env.reset()
    for _ in range(t_max):
        
        actions_probs = predict_probs([s])[0]
        a = np.random.choice(range(env.action_space.n), p=actions_probs)
        next_s, r, done, _ = env.step(a)

        states.append(s)
        actions.append(a)
        rewars.append(r)

        s = next_s
        if done:
            break
    
    return states, actions, rewars


def get_cumulative_rewards(rewards, gamma=0.99):
    '''
    G return
    '''

    _rewards = [0 for i in range(len(rewards)+1)]

    for i, r in enumerate(rewards[::-1]):
        _rewards[i+1] = gamma * _rewards[i] + r

    return _rewards[1:][::-1]


optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train_on_session(states, actions, rewards, gamma=0.99, entropy_coef=1e-2):
    '''
    '''
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    cumulative_rewards = torch.tensor(get_cumulative_rewards(rewards, gamma), dtype=torch.float32)

    probs = model(states)
    log_probs = probs.log_softmax(dim=-1)
    log_probs_for_actions = log_probs[range(len(actions)), actions]

    entropy = -entropy_coef * log_probs_for_actions 
    loss = (entropy * cumulative_rewards).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return np.sum(rewards)



for _ in range(100):
    rewards = [train_on_session(*generate_session()) for _ in range(100)]

    print(np.mean(rewards))

    if np.mean(rewards) > 500:
        print('done.')
        break
