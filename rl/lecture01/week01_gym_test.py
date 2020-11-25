import gym
from gym import envs

def print_gym_info():
    '''
    '''
    envids = [spec.id for spec in envs.registry.all()]

    print(sorted(envids))
    print('Total nums: ', len(envids))


def print_envs_info():
    '''
    '''
    env = gym.make('CartPole-v0')
    print(env.action_space)
    print(env.observation_space)


class RandomAgent(object):
    
    def __init__(self, init_action_space):
        '''
        '''
        self.action_space = init_action_space
    
    def act(self, observation, reward, done, info):
        '''
        '''
        return self.action_space.sample()


def main():
    '''
    '''

    env = gym.make('MountainCar-v0')
    env = gym.wrappers.Monitor(env, 'videos')

    agent = RandomAgent(env.action_space)

    episode_count = 10
    reward = 0
    done = False

    for _ in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done, '')
            ob, reward, done, _ = env.step(action)

            if done:
                break
    
    env.close()


"""
env = gym.make('MountainCar-v0') #('CartPole-v0')

# print(envs.registry.all())

print(env.action_space)
print(env.observation_space)


for i_episode in range(20):
    observation = env.reset()
    for t in range(1000):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation, reward, done, info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

"""


if __name__ == '__main__':
    '''
    '''

    # print_gym_info()
    # print_envs_info()
    main()