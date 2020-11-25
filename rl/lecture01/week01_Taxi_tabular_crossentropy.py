import gym
import numpy as np 
import matplotlib.pyplot as plt
import itertools
import IPython.display as display


name ='Taxi-v3'

env = gym.make(name)
env.reset()
# env.render()

n_states = env.observation_space.n
n_actions = env.action_space.n 

print('number of states space: ', n_states)
print('number of action space: ', n_actions)


def init_tabular_policy(n_states, n_actions):
    '''
    '''
    policy = np.ones((n_states, n_actions)) / n_actions

    assert type(policy) in (np.ndarray, np.matrix)
    assert np.allclose(policy, 1. / n_actions)
    assert np.allclose(np.sum(policy, axis=1), 1)

    return policy


def generate_session(policy, env=env, t_max = 10**4):
    '''
    play game util end or t_max ticks
    '''

    states = []
    actions = []
    reward = 0.

    s = env.reset()

    for _ in range(t_max):

        a = np.random.choice(range(n_actions), p=policy[s])
        s_next, r, done, _ = env.step(a)
        # env.render()
        
        states.append(s)
        actions.append(a)
        reward += r

        s = s_next

        if done:
            break
    
    return states, actions, reward


def select_elites(states, actions, rewards, percentile=50):
    '''
    states[session_i][t_i]
    actions[session_i][t_i]
    rewards[session_i]
    '''
    rewards = np.array(rewards)
    r_threshold = np.percentile(rewards, percentile)

    # idx = [i for i, r in enumerate(rewards) if r > r_threshold]
    # idx = (np.array(rewards) > r_threshold).nonzero()[0]
    idx = np.where(rewards >= r_threshold)[0]

    elite_states = [s for i in idx for s in states[i]]
    elite_actions = [a for i in idx for a in actions[i]]

    # elite_states = list(itertools.chain(*[states[i] for i in idx]))
    # elite_actions = list(itertools.chain(*[actions[i] for i in idx]))

    return elite_states, elite_actions


def update_policy(policy, elite_states, elite_actions):
    '''
    '''
    new_policy = np.ones_like(policy) / n_actions
    
    elite_states = np.array(elite_states)
    elite_actions = np.array(elite_actions)

    for s_i in range(n_states):
        # 
        sa_count = np.zeros((n_actions, ))
        idx = ((elite_states == s_i).nonzero())[0]
        for i in idx:
            sa_count[elite_actions[i]] += 1

        # new_policy[s_i] = sa_count / len(idx)
        new_policy[s_i] = (sa_count + 1) / (len(idx) + n_actions)

        # if len(idx) == 0:
        #     prob = 1. / n_actions
        # else:
        #     prob = sa_count / len(idx) + 1. / n_actions
        #     prob = prob / sum(prob)
        # new_policy[s_i] = prob

    assert np.all(new_policy > 0), ''
    assert np.allclose(new_policy.sum(axis=-1), 1), ''

    return new_policy


        
def show_progress(rewards, log, percentile, reward_range=[-990, +10]):
    '''

    '''
    mean_reward = np.mean(rewards)
    r_threshold = np.percentile(rewards, percentile)
    log.append([mean_reward, r_threshold])

    # plt.cla()
    display.clear_output(True)
    
    plt.figure(figsize=[8, 4])
    plt.subplot(1, 2, 1)
    plt.plot(list(zip(*log))[0], label='mean reward')
    plt.plot(list(zip(*log))[1], label='reward threshold')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.hist(rewards, range=reward_range)
    plt.vlines([np.percentile(rewards, percentile)], [0], [100], label='percentile', colors='red')
    plt.legend()
    plt.grid()

    # plt.draw()
    plt.show()

    print('mean_reward: ', mean_reward)


def show_analysis(policy, VIS=True):
    '''
    '''
    sample_rewards = [generate_session(policy)[-1] for _ in range(300)]

    plt.hist(sample_rewards, bins=20)
    plt.vlines([np.percentile(sample_rewards, 50)], [0], [100], label='50 percentile', color='red')
    plt.vlines([np.percentile(sample_rewards, 90)], [0], [100], label='90 percentile', color='blue')
    plt.legend()

    if VIS:
        plt.show()


def train(epoches=100, n_session=500, percentile=50, learning_rate=0.5):
    '''
    '''
    policy = init_tabular_policy(n_states, n_actions)
    log = []
    for _ in range(epoches):
        sessions = [generate_session(policy) for _ in range(n_session)]
        states, actions, rewards = zip(*sessions)

        elite_states, elite_actions = select_elites(states, actions, rewards, percentile=percentile)

        new_policy = update_policy(policy, elite_states, elite_actions)
        policy = learning_rate * new_policy + (1 - learning_rate) * policy

        # show_progress(rewards, log, percentile)
        print(np.mean(rewards))

    return policy

# policy = init_tabular_policy(n_states, n_actions)
# s, a, r = generate_session(policy)
# # show_analysis(policy, VIS=False)

# states = [[1, 2, 3], [4, 2, 0, 2], [3, 1]]
# actions = [[0, 2, 4], [3, 2, 0, 1], [3, 3]]
# rewards = [3, 4, 5]
# test0 = select_elites(states, actions, rewards, percentile=0)
# print(test0)

# elite_states = [1, 2, 3, 4, 2, 0, 2, 3, 1]
# elite_actions = [0, 2, 4, 3, 2, 0, 1, 3, 3]
# new_policy = update_policy(policy, elite_states, elite_actions)
# print(new_policy[:4, :])


policy = train(epoches=100, n_session=300)


env = gym.wrappers.Monitor(gym.make(name), directory='./videos', force=True)
session = [generate_session(policy, env=env) for _ in range(10)]
env.close()

