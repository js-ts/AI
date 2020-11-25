import gym
from gym.utils.play import play
from gym.core import ObservationWrapper
from gym.spaces import Box

import numpy as np 
import random
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image

import atari_wrappers
import framebuffer

env = gym.make('BreakoutNoFrameskip-v4')
# play(env, fps=30, zoom=5)


class PreprocessAtariObs(ObservationWrapper):
    def __init__(self, env):
        '''
        '''
        super(PreprocessAtariObs, self).__init__(env)

        self.img_size = (1, 64, 64)
        self.observation_space = Box(0.0, 1.0, shape=self.img_size)


    def observation(self, img):
        '''
        '''
        h, w, _ = img.shape
        im = Image.fromarray(img).convert('L')
        im = im.crop((10, 20, w-10, h-10))
        im = im.resize((64, 64))
        im = np.array(im, dtype=np.float32) / 255.
        
        return im[None]


# env = gym.make('BreakoutNoFrameskip-v4')
# env = PreprocessAtariObs(env)
# print(env.observation_space.shape)
# s = env.reset()
# print(s.shape)


def PrimaryAtariWrap(env, clip_rewards=True):
    '''
    '''
    assert 'NoFrameskip' in env.spec.id
    env = atari_wrappers.MaxAndSkipEnv(env, skip=4)
    env = atari_wrappers.EpisodicLifeEnv(env)
    env = atari_wrappers.FireResetEnv(env)
    if clip_rewards:
        env = atari_wrappers.ClipRewardEnv(env)
    env = PreprocessAtariObs(env)
    
    return env


def make_env(clip_rewards=True, seed=None):
    '''
    '''

    env = gym.make('BreakoutNoFrameskip-v4')
    if seed is not None:
        env.seed(seed)
    env = PrimaryAtariWrap(env)
    env = framebuffer.FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env




class DQNAgent(nn.Module):
    '''
    '''
    def __init__(self, state_shape, n_actions, epsilon=0.5):
        '''
        '''
        super(DQNAgent, self).__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=state_shape[0], out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(), 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.dense = nn.Sequential(nn.Linear(64 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, n_actions))


    def forward(self, state):
        '''
        '''
        qvalues = self.backbone(state)
        qvalues = self.dense(qvalues.view(qvalues.shape[0], -1))

        return qvalues


    def get_qvalues(self, states):
        '''
        '''
        device = next(self.parameters()).device
        states = torch.tensor(states, device=device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.cpu().data.numpy()
    

    def sample_actions(self, qvalues):
        '''
        '''
        epsilon = self.epsilon
        n, c = qvalues.shape

        if random.random() < epsilon:
            return np.random.choice(c, n)
        
        return qvalues.argmax(axis=-1)



class _ReplayBuffer(object):

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
        idxes = [
            random.randint(0, len(self._storage) - 1)
            for _ in range(batch_size)
        ]
        return self._encode_sample(idxes)




def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    '''
    '''
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            a = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            next_s, r, done, _ = env.step(a)
            s = next_s
            reward += r 
            if done:
                break
            
        rewards.append(reward)

    return np.mean(rewards)



def play_and_record(s, agent, env, exp_replay, n_step=1):
    '''
    '''
    sum_rewards = 0

    for _ in range(n_step):

        qvalues = agent.get_qvalues([s])
        a = agent.sample_actions(qvalues)
        next_s, r, done, _ = env.step(a)

        exp_replay.add(s, a, r, next_s, done)

        sum_rewards += r
        s = next_s

        if done:
            s = env.reset()

    return sum_rewards, s


def compute_td_loss(states, actions, rewards, next_states, dones, agent, target_network, gamma=0.9):
    '''
    '''
    device = 'cpu'
    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.bool, device=device)

    predicted_qvalues = agent(states)
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions.view(-1)]

    target_qvalues = target_network(next_states)
    target_states_values, _ = target_qvalues.max(dim=-1)
    target_qvalues_for_actions = rewards + gamma * target_states_values
    target_qvalues_for_actions = torch.where(dones, rewards, target_qvalues_for_actions)

    loss = (predicted_qvalues_for_actions - target_qvalues_for_actions).pow(2).mean()

    return loss


env = make_env()
s = env.reset()
n_actions = env.action_space.n
state_shape = env.observation_space.shape 

agent = DQNAgent(state_shape, n_actions)
opt = optim.Adam(agent.parameters(), lr=1e-4)

qvalues = agent.get_qvalues([s])
actions = agent.sample_actions(qvalues)
evaluate(env, agent, 1)

exp_replay = ReplayBuffer(2000)
state = env.reset()
play_and_record(state, agent, env, exp_replay, n_step=1000)

target_network = DQNAgent(state_shape, n_actions)
target_network.load_state_dict(agent.state_dict())

obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(10)
loss = compute_td_loss(obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch,agent, target_network,gamma=0.99)


total_steps = 3 * 10 ** 4
decay_step = 10 ** 4
refresh_target_network_freq = 100
init_epsilon = 1. 
final_epsilon = 0.1
timesteps_per_epoch = 1
max_grad_norm = 50

s = env.reset()

for step in range(total_steps+1):
    
    agent.epsilon = init_epsilon - (init_epsilon - final_epsilon) / total_steps * step
    _, next_s = play_and_record(s, agent, env, exp_replay, timesteps_per_epoch)

    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(8)
    loss = compute_td_loss(obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch, agent, target_network, gamma=0.99)

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    opt.step()

    if step % refresh_target_network_freq == 0:
        target_network.load_state_dict(agent.state_dict())
        _r = evaluate(env, agent, n_games=1, greedy=True)
        print(_r)

