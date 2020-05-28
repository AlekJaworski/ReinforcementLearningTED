import gym

env_names = ['CartPole-v0', 'MountainCar-v0', 'MsPacman-v0', 'Hopper-v2']

env = gym.make(env_names[0])

env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())  # take a random action
env.close()
