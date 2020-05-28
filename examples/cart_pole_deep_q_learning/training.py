from time import sleep

import gym
import numpy as np
import tensorflow as tf


# input_dimensions = 3
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(16, activation='relu', input_shape=(input_dimensions,)),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(2),
#     tf.keras.layers.Softmax()
# ])


class CartPoleAgent:
    def __init__(self, environment):
        self.state_dim = environment.observation_space.shape
        self.action_dim = environment.action_space.n
        self.q_network: QNetwork = QNetwork(self.state_dim, self.action_dim)
        self.gamma = 0.95
        self.epsilon = 0.99

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return env.action_space.sample()
        q_vals = self.q_network.evaluate(state)
        action = np.argmax(q_vals)
        return action

    def train(self, state, action, next_state, reward, done):
        q_next_state = self.q_network.evaluate(next_state)
        q_next_state = (1 - done) * q_next_state  # if done then set to 0
        q_target = reward + self.gamma * np.max(q_next_state)
        one_hot_action = tf.one_hot(action, depth=self.action_dim)

        state_dummy = state.reshape(1, 4)
        with tf.GradientTape() as tape:
            q_values = self.q_network.model(state_dummy)
            extracted = tf.reduce_sum(tf.multiply(q_values, one_hot_action))
            loss = (extracted - q_target) ** 2
            # print(f'loss: {loss}')

        grads = tape.gradient(loss, self.q_network.model.trainable_variables)
        self.q_network.minimizer.apply_gradients(list(zip(grads, self.q_network.model.trainable_variables)))

        if done:
            self.epsilon = max(0.1, 0.99 * self.epsilon)


class QNetwork:
    def __init__(self, state_dimension, action_dimension):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(120, activation='relu', input_shape=state_dimension),
            tf.keras.layers.Dense(action_dimension)
        ])
        self.model: tf.keras.Model = model
        self.minimizer = tf.optimizers.Adam(learning_rate=1e-3)

    def evaluate(self, state):
        arg = state.reshape(1, 4)
        return self.model.predict(arg)


env_names = ['CartPole-v0', 'MountainCar-v0', 'MsPacman-v0', 'Hopper-v2']

env = gym.make(env_names[0])

env.reset()

agent = CartPoleAgent(env)
num_episodes = 500
for ep in range(num_episodes):
    total_reward = 0
    done = False
    state = env.reset()
    while not done:
        action = agent.get_action(state)
        # print(f'action: {action}')
        next_state, reward, done, info = env.step(action)
        agent.train(state, action, next_state, reward, done)
        env.render()
        total_reward += reward
        state = next_state
    # sleep(0.5)

    print(f'episode: {ep}, total reward: {total_reward}')
