from abc import ABC
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.initializers import Constant, Zeros
from tensorflow.python.keras import layers, Model
from agents.agent import Agent
from tensorflow import keras


class A2CActor(keras.Model):

    def __init__(self, policy, independent_log_stds, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.policy = policy
        self.independent_log_stds = independent_log_stds

    def call(self, inputs, training=None, mask=None):
        if self.independent_log_stds is None:
            means, std_devs = self.policy(inputs)
            std_devs = tf.clip_by_value(tf.math.exp(std_devs), 1e-3, 50)
        else:
            means = self.policy(inputs)
            std_devs = [tf.clip_by_value(tf.math.exp(logstd), 1e-3, 50) for logstd in self.independent_log_stds]
            std_devs = tf.stack(std_devs)
        return means, std_devs


class A2CNStepAheadAgent(Agent):

    @staticmethod
    def get_algo():
        return 'A2C_N_STEP_AHEAD'

    def __init__(self, *agent_params, n_steps=5, entropy_coeff=1e-2, log_std_init=-0.5, std_state_dependent=False):

        self.log_std_init = log_std_init
        self.std_state_dependent = std_state_dependent
        self.entropy_coefficient = entropy_coeff

        super().__init__(*agent_params)

        self.n_steps = n_steps

        # indices
        self.finished = 0
        self.dones = 0

    def get_actor(self):
        input_layer = keras.layers.Input(shape=self.state_shape)
        hidden = input_layer

        for i in range(int(self.n_layers_actor) if not self.std_state_dependent else int(self.n_layers_actor) - 1):
            hidden = keras.layers.Dense(int(self.units_per_layer_actor), activation=self.activation_actor)(hidden)

        if self.std_state_dependent:
            hidden_m = keras.layers.Dense(int(self.units_per_layer_actor), activation=self.activation_actor)(hidden)
        else:
            hidden_m = hidden

        means = keras.layers.Dense(self.action_shape[0], activation='tanh')(hidden_m)
        independent_logstds = None

        if self.std_state_dependent:
            hidden_s = keras.layers.Dense(int(self.units_per_layer_actor), activation=self.activation_actor)(hidden)
            dependent_logstds = keras.layers.Dense(self.action_shape[0],
                                                   kernel_initializer=keras.initializers.Zeros(),
                                                   bias_initializer=keras.initializers.Constant(self.log_std_init))(
                hidden_s)
            policy = keras.Model(inputs=input_layer, outputs=[means, dependent_logstds])

        else:
            independent_logstds = []
            policy = keras.Model(inputs=input_layer, outputs=means)

            for i in range(self.action_shape[0]):
                independent_logstds.append(policy.add_weight(name=f'logstd_{i}',
                                                             shape=(),
                                                             initializer=keras.initializers.Constant(
                                                                 self.log_std_init),
                                                             trainable=True))

        return A2CActor(policy, independent_logstds)

    def get_critic(self):
        # State as input
        input_layer = keras.layers.Input(shape=self.state_shape)
        hidden = input_layer

        for i in range(int(self.n_layers_critic)):
            hidden = keras.layers.Dense(int(self.units_per_layer_critic), activation=self.activation_critic)(hidden)

        outputs = keras.layers.Dense(1)(hidden)

        # Outputs single value for give state-action
        model = tf.keras.Model(input_layer, outputs)

        return model

    def act(self, state):
        means, std_devs = self.actor(state)
        action = np.random.normal(loc=means, scale=std_devs)
        return action.flatten()

    def n_step_reward(self, curr_state, done, reward, env, agent):

        state_id = env.save_state()
        curr_gamma = self.gamma
        n_step_reward = reward
        actual_steps = 0

        for i in range(self.n_steps - 1):

            if done:
                break

            action = agent.act(tf.convert_to_tensor([curr_state], dtype='float32'))
            curr_state, reward, done, _, _ = env.step(action)

            if env.unwrapped.spec.id == "PandaPushDense-v3":
                reward -= 0.5 * np.linalg.norm(curr_state['observation'][:3] - curr_state['achieved_goal'])
                reward -= 0.1 * np.linalg.norm(action) ** 2

            curr_state = np.concatenate([curr_state['observation'], curr_state['desired_goal']], dtype=np.float32)

            n_step_reward += curr_gamma * tf.convert_to_tensor(reward, dtype='float32')

            curr_gamma *= self.gamma

            actual_steps += 1

        env.restore_state(state_id)
        env.remove_state(state_id)

        return n_step_reward, done, tf.convert_to_tensor([curr_state]), actual_steps

    def __run_episode(self, env, episode):
        state, info = env.reset()

        state = np.concatenate([state['observation'], state['desired_goal']], dtype=np.float32)

        state = tf.convert_to_tensor([state], dtype='float32')

        total_reward = 0
        done = False
        num_step = 0

        while not done and num_step < 50:
            action = self.act(state)
            next_state, reward, done, truncated, info = env.step(action)

            if done and num_step <= 2:
                return None

            if env.unwrapped.spec.id == "PandaPushDense-v3":
                reward -= 0.5 * np.linalg.norm(next_state['observation'][:3] - next_state['achieved_goal'])

                reward -= 0.1 * np.linalg.norm(action) ** 2

            next_state = np.concatenate([next_state['observation'], next_state['desired_goal']], dtype=np.float32)

            action = tf.convert_to_tensor(action, dtype='float32')
            reward = tf.convert_to_tensor(reward, dtype='float32')

            n_steps_reward, done_cur, n_step_state, actual_steps = self.n_step_reward(next_state, done, reward,
                                                                                      env, self)

            next_state = tf.convert_to_tensor([next_state], dtype='float32')
            n_step_state = tf.convert_to_tensor(n_step_state, dtype='float32')
            n_steps_reward = tf.convert_to_tensor(n_steps_reward, dtype='float32')

            self.__train(state, action, reward, next_state, n_steps_reward, done_cur, n_step_state, actual_steps + 1)
            state = next_state
            total_reward += reward
            num_step += 1

        if not done:
            self.finished = episode
        else:
            self.dones += 1

        return total_reward

    @tf.function
    def __train(self, state, action, reward, next_state, n_steps_reward, done_cur, n_step_state, n_steps,
                entropy_coeff=0.01, grad_clip=-1):
        with tf.GradientTape(persistent=True) as tape:

            means, std_devs = self.actor(state)
            value = self.critic(state)

            action_prob = tf.exp(-0.5 * tf.square((action - means) / std_devs)) / (std_devs * tf.sqrt(2. * np.pi))
            action_prob = tf.reduce_prod(action_prob, axis=1, keepdims=True)

            n_step_value = (1 - done_cur) * self.critic(n_step_state)

            advantage = n_steps_reward + (self.gamma ** n_steps) * n_step_value - value

            # Compute the entropy of the current policy
            entropy = 0.5 * (tf.math.log(2 * np.pi * std_devs ** 2) + 1)
            entropy = tf.reduce_sum(entropy, keepdims=True)

            # Compute the actor loss
            actor_loss = -tf.reduce_mean(
                tf.math.log(tf.maximum(action_prob, 5e-6)) * tf.stop_gradient(advantage) + entropy_coeff * entropy)

            # Compute the critic loss 
            critic_loss = tf.reduce_mean(
                tf.square(tf.stop_gradient(n_steps_reward + (self.gamma ** n_steps) * n_step_value) - value))

        # Compute and apply the gradients for the actor loss
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

        # Clip the gradients
        if grad_clip != -1:
            actor_grads, _ = tf.clip_by_global_norm(actor_grads, grad_clip)

        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Compute and apply the gradients for the critic loss
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

        # Clip the gradients
        if grad_clip != -1:
            critic_grads, _ = tf.clip_by_global_norm(critic_grads, grad_clip)

        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Clear the tape to free up resources
        del tape

    def train(self, state, action, reward, next_state, done):
        return

    def train_agent(self, env, episodes, verbose=0):
        env = env.env  # block the truncated 

        return super().train_agent(env, episodes, verbose)
