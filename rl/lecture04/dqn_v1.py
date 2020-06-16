import numpy as np
import gym
import random
import torch
import torch.nn as nn
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQNAgent(nn.Module):
    '''
    '''
    def __init__(self, state_shape, n_actions, epsilon=0):
        super(DQNAgent, self).__init__()

        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        state_dim = np.prod(state_shape)
        hidden_dim = 64
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, states):
        '''
        '''
        qvalues = self.layers(states)
        return qvalues

    def get_qvalues(self, states):
        '''
        '''
        device = next(self.parameters()).device
        states = torch.tensor(states, dtype=torch.float32, device=device)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()
    
    def sample_actions(self, qvalues):
        '''
        '''
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        rand_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        exporation = np.random.choice([0, 1], size=batch_size, p=[1-epsilon, epsilon])

        return np.where(exporation, rand_actions, best_actions)



def evaluate(env, agent, n_games=1, greedy=False, t_max=1000):
    '''
    '''
    rewards = []
    for _ in range(n_games):
        
        s = env.reset()
        reward = 0

        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            a = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0] 
            nexts, r, done, _ = env.step(a)
            reward += r 
            s = nexts
            if done:
                env.reset()
                break

        rewards.append(reward)
    
    return np.mean(rewards)


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return (
            np.array(obses_t),
            np.array(actions),
            np.array(rewards),
            np.array(obses_tp1),
            np.array(dones)
        )

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [
            random.randint(0, len(self._storage) - 1)
            for _ in range(batch_size)
        ]
        return self._encode_sample(idxes)


def play_and_record(initial_sate, agent, env, exp_replay, n_steps=1):
    '''
    '''
    s = initial_sate
    sum_rewards = 0

    for _ in range(n_steps):
        
        qvalues = agent.get_qvalues([s])
        a = agent.sample_actions(qvalues)[0]

        next_s, r, done, _ = env.step(a)

        exp_replay.add(s, a, r, next_s, done)

        sum_rewards += r 
        s = next_s

        if done:
            s = env.reset()

    return sum_rewards, s

def compute_td_loss(states, actions, rewards, next_states, is_done, agent, target_network, gamma=0.99, device=device):
    '''
    '''
    states = torch.tensor(states, device=device, dtype=torch.float32)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
    next_states = torch.tensor(next_states, device=device, dtype=torch.float32)
    is_done = torch.tensor(is_done, device=device, dtype=torch.bool)

    predicted_qvalues = agent(states)
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]

    predicted_next_qvalues = target_network(next_states)
    next_states_values, _ = torch.max(predicted_next_qvalues, dim=-1)
    target_qvalues_for_action = rewards + gamma * next_states_values
    target_qvalues_for_action = torch.where(is_done, rewards, target_qvalues_for_action)

    loss = torch.pow(predicted_qvalues_for_actions - target_qvalues_for_action, 2).mean()

    return loss


# obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(10)
# loss = compute_td_loss(obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch, agent, target_network, gamma=0.99)
# loss.backward()  

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

env = gym.make('CartPole-v1').unwrapped
env.seed(seed)

env.reset()
state_shape = env.observation_space.shape
n_actions = env.action_space.n 

agent = DQNAgent(state_shape, n_actions)

target_network = DQNAgent(agent.state_shape, agent.n_actions, epsilon=0.5).to(device)
target_network.load_state_dict(agent.state_dict())

opt = torch.optim.Adam(agent.parameters(), lr=1e-4)

exp_replay = ReplayBuffer(2000)
state = env.reset()
play_and_record(state, agent, env, exp_replay, n_steps=100)

batch_size = 32
total_steps = 40000
init_epsilon = 1
final_epsilon = 0.05
refresh_target_network_freq = 100

max_grad_norm = 5000

s = env.reset()
for step in range(total_steps + 1):
    
    # agent.epsilon = init_epsilon - (init_epsilon - final_epsilon) / total_steps * step
    if init_epsilon * 0.999 < final_epsilon:
        agent.epsilon = final_epsilon
    else:
        init_epsilon *= 0.999
        agent.epsilon = init_epsilon

    _, s = play_and_record(s, agent, env, exp_replay, 1)
    # qvalues = agent.get_qvalues([s])
    # a = agent.sample_actions(qvalues)[0]
    # next_s, r, done, _ = env.step(a)
    # exp_replay.add(s, a, r, next_s, done)
    # s = next_s
    # if done:
    #     s = env.reset()

    states, actions, rewards, next_states, done = exp_replay.sample(batch_size)
    loss = compute_td_loss(states, actions, rewards, next_states, done, agent, target_network)

    opt.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    opt.step()

    if step % refresh_target_network_freq == 0:
        target_network.load_state_dict(agent.state_dict())
    
    if step % 100 == 0:
        _r = evaluate(env, agent, n_games=20, greedy=True)
        print(step, loss.item(), _r)
        pass
    
