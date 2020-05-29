from collections import deque
import random

import gym
import numpy as np
import tensorflow as tf

EPSILON_DECAY = 0.99
EPSILON_MIN = 0.1
BATCH_SIZE = 50


class CartPoleAgent:
    def __init__(self, environment):
        self.state_dim = environment.observation_space.shape
        self.action_dim = environment.action_space.n
        self.q_network: QNetwork = QNetwork(self.state_dim, self.action_dim)
        self.gamma = 0.95
        self.epsilon = 0.99
        self.replay_buffer = ReplayBuffer(maxlen=10000)
        self.buffer_size = BATCH_SIZE

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return env.action_space.sample()
        q_vals = self.q_network.evaluate(np.array([state]))
        action = np.argmax(q_vals)
        return action

    def train_with_replay(self, state, action, next_state, reward, done):
        self.replay_buffer.add((state, action, next_state, reward, done))

        samples = self.replay_buffer.sample(min(self.buffer_size, BATCH_SIZE))
        states, actions, next_states, rewards, dones = samples
        states.append(state)
        states = np.array(states)

        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        dones.append(done)

        q_next_states = self.q_network.evaluate(next_states)
        q_next_states[dones] = np.zeros(self.action_dim)  # if done then set to 0
        q_target = reward + self.gamma * np.max(q_next_states, axis=1)
        one_hot_action = tf.one_hot(actions, depth=self.action_dim)

        with tf.GradientTape() as tape:
            q_values = self.q_network.model(states)
            extracted = tf.reduce_sum(tf.multiply(q_values, one_hot_action), axis=1)
            loss = tf.reduce_sum((extracted - q_target) ** 2)

        grads = tape.gradient(loss, self.q_network.model.trainable_variables)
        self.q_network.minimizer.apply_gradients(list(zip(grads, self.q_network.model.trainable_variables)))

        if done:
            self.epsilon = max(EPSILON_MIN, EPSILON_DECAY * self.epsilon)

    def train(self, state, action, next_state, reward, done):

        q_next_states = self.q_network.evaluate([next_state])
        q_next_states[[done]] = np.zeros(self.action_dim)  # if done then set to 0
        q_target = reward + self.gamma * np.max(q_next_states, axis=1)
        one_hot_action = tf.one_hot([action], depth=self.action_dim)

        with tf.GradientTape() as tape:
            q_values = self.q_network.model(np.array([state]))
            extracted = tf.reduce_sum(tf.multiply(q_values, one_hot_action), axis=1)
            loss = tf.reduce_sum((extracted - q_target) ** 2)

        grads = tape.gradient(loss, self.q_network.model.trainable_variables)
        self.q_network.minimizer.apply_gradients(list(zip(grads, self.q_network.model.trainable_variables)))

        if done:
            self.epsilon = max(EPSILON_MIN, EPSILON_DECAY * self.epsilon)


class ReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)

        samples = random.choices(self.buffer, k=sample_size)

        return list(map(list, zip(*samples)))


class QNetwork:
    def __init__(self, state_dimension, action_dimension):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(120, activation='relu', input_shape=state_dimension),
            tf.keras.layers.Dense(action_dimension)
        ])
        self.model: tf.keras.Model = model
        self.minimizer = tf.optimizers.Adam(learning_rate=1e-3)

    def evaluate(self, states):
        arg = np.array(states)
        return self.model.predict(arg)


if __name__ == '__main__':

    env_names = ['CartPole-v0', 'MountainCar-v0', 'MsPacman-v0', 'Hopper-v2', 'Acrobot-v1']

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
            next_state, reward, done, info = env.step(action)
            # agent.train(state, action, next_state, reward, done)
            agent.train_with_replay(state, action, next_state, reward, done)
            env.render()
            total_reward += reward
            state = next_state

        print(f'episode: {ep}, total reward: {total_reward}')
