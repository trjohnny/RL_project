import logging
from abc import abstractmethod
from abc import ABC
import tensorflow as tf
import numpy as np


class Agent(ABC):
    def __init__(self,
                 n_states,
                 n_actions,
                 lower_bound,
                 upper_bound,
                 gamma=.85,
                 actor_learning_rate=1e-4,
                 critic_learning_rate=5e-4,
                 n_layers_actor=3,
                 n_layers_critic=2,
                 units_per_layer_actor=80,
                 units_per_layer_critic=64):

        self.gamma = gamma
        self.state_shape = (n_states,)
        self.action_shape = (n_actions,)
        self.n_layers_actor = n_layers_actor
        self.n_layers_critic = n_layers_critic
        self.units_per_layer_actor = units_per_layer_actor
        self.units_per_layer_critic = units_per_layer_critic
        self.activation_actor = 'relu'
        self.activation_critic = 'relu'

        self.actor = self.get_actor()
        self.critic = self.get_critic()

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_learning_rate)

        # indices
        self.finished = 0
        self.dones = 0

    @abstractmethod
    def get_actor(self):
        pass

    @staticmethod
    @abstractmethod
    def get_algo():
        pass

    @abstractmethod
    def get_critic(self):
        pass

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def train(self, state, action, reward, next_state, done):
        pass

    def __run_episode(self, env, episode): 
        state, info = env.reset()
        state = np.concatenate([state['observation'], state['desired_goal']], dtype=np.float32)
 
        state_tensor = tf.convert_to_tensor([state], dtype='float32')

        total_reward = 0
        done = False
        truncated = False
        num_step = 0

        while not done and not truncated:
            action = self.act(state_tensor)

            next_state, reward, done, truncated, info = env.step(action)

            if env.unwrapped.spec.id == "PandaPushDense-v3":
                reward -= 0.5 * np.linalg.norm(next_state['observation'][:3] - next_state['achieved_goal'])
                reward -= 0.1 * np.linalg.norm(action) ** 2

            reward *= 10

            next_state = np.concatenate([next_state['observation'], next_state['desired_goal']], dtype=np.float32)

            self.train(state, action, reward, next_state, done)

            state = next_state
            state_tensor = tf.convert_to_tensor([state], dtype='float32')
            total_reward += reward
            num_step += 1

        if not done:
            self.finished = episode
        else:
            self.dones += 1

        return total_reward

    def train_agent(self, env, episodes, verbose=1):
        rewards = []
        mod = episodes - 1
        if verbose == 1:
            mod = 100
        elif verbose == 2:
            mod = 10
        elif verbose == 3:
            mod = 1

        for episode in range(1, episodes + 1):
 
            reward = self.__run_episode(env, episode)

            if reward is not None:
                rewards.append(reward)
            else:
                continue

            # Print the reward for each episode
            if episode % mod == 0:
                print(f'Episode {episode}: Reward: {reward:.2f}')

        return rewards
