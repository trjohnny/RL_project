from abc import ABC
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.initializers import Constant, Zeros
from tensorflow.python.keras import layers, Model
from agent import Agent


class A2CDiscreteActor(Model):
    def __init__(self, policy_model, log_std, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_model = policy_model
        self.log_std = log_std

    def get_config(self):
        super().get_config()
        pass

    def _serialize_to_tensors(self):
        super()._serialize_to_tensors()
        pass

    def _restore_from_tensors(self, restored_tensors):
        super()._restore_from_tensors(restored_tensors)
        pass

    def call(self, inputs, training=None, mask=None):
        means = self.policy(inputs)
        std_devs = [tf.clip_by_value(tf.math.exp(log_std), 1e-3, 50) for log_std in self.log_std]
        std_devs = tf.stack(std_devs)
        return means, std_devs


class A2CDiscreteAgent(Agent):
    def __init__(self, *agent_params, n_steps=5, entropy_coeff=1e-2):
        super().__init__(*agent_params)

        self.actor = self.__get_actor()
        self.critic = self.__get_critic()
        self.n_steps = n_steps
        self.entropy_coefficient = entropy_coeff

        # indices
        self.finished = 0
        self.dones = 0

    def __get_actor(self):
        self.input_layer = layers.Input(shape=self.state_shape)

        self.hidden_layer = layers.Dense(80, activation='relu')(self.input_layer)
        self.hidden_layer = layers.Dense(80, activation='relu')(self.hidden_layer)
        self.hidden_layer_m = layers.Dense(80, activation='relu')(self.hidden_layer)

        self.means = layers.Dense(self.action_shape[0], activation='tanh')(self.hidden_layer_m)

        self.logstds = []
        for i in range(self.action_shape[0]):
            self.logstds.append(self.add_weight(name=f'logstd_{i}',
                                                    shape=(),
                                                    initializer=Constant(-0.75),
                                                    trainable=True))
        self.policy = Model(inputs=self.input_layer, outputs=self.means)
        return A2CDiscreteActor(self.policy)

    def __get_critic(self):
        # State as input
        self.input_layer = layers.Input(shape=self.state_shape)

        self.hidden_layer = layers.Dense(80, activation='relu')(self.input_layer)
        self.hidden_layer = layers.Dense(80, activation='relu')(self.hidden_layer)

        self.output_layer = layers.Dense(self.action_shape[0])(self.hidden_layer)

        self.value_function = Model(inputs=self.input_layer, outputs=self.output_layer)
        return self.value_function


    def act(self, state):
        means, std_devs = self.actor(state)
        action = np.random.normal(loc=means, scale=std_devs)
        return action.flatten()

    def n_step_reward(self, n_steps, curr_state, done, reward, env, agent):

        state_id = env.save_state()
        curr_gamma = self.gamma
        n_step_reward = reward
        actual_steps = 0

        for i in range(n_steps - 1):

            if done:
                break

            action = agent.act(tf.convert_to_tensor([curr_state], dtype='float32'))
            curr_state, reward, done, _, _ = env.step(action)

            reward -= 0.5 * np.linalg.norm(curr_state['observation'][:3] - curr_state['achieved_goal'])
            reward -= 0.1 * np.linalg.norm(action) ** 2

            curr_state = np.concatenate([curr_state['observation'][:9], curr_state['desired_goal']], dtype=np.float32)

            n_step_reward += curr_gamma * tf.convert_to_tensor(reward, dtype='float32')

            curr_gamma *= self.gamma

            actual_steps += 1

        env.restore_state(state_id)
        env.remove_state(state_id)

        return n_step_reward, done, tf.convert_to_tensor([curr_state]), actual_steps

    @tf.function
    def __train(self, state, action, reward, next_state, done, grad_clip, entropy_coeff=0.00):
        action_index = int(((action+1)*self.discrete_values)/2)
        with tf.GradientTape(persistent=True) as tape:

            pre_means = self.actor(state)  # array of tensors
            # pre_means is a vector of tensors, each tensor is a vector of probabilities
            means, std_devs = self.calculate_means_stds(pre_means)
            value = self.critic(state)
            next_value = self.critic(next_state)

            indices = tf.stack([tf.range(pre_means.shape[0]), action_index], axis=1)
            action_probability = tf.gather_nd(pre_means, indices)

            advantage = reward + (1 - done) * self.gamma * next_value - value

            # Compute the entropy of the current policy
            entropy = -tf.reduce_sum(pre_means * tf.math.log(pre_means + tf.keras.backend.epsilon()), axis=-1)

            # Compute the actor loss
            actor_loss = -tf.reduce_mean(tf.math.log(tf.maximum(action_probability, 5e-6)) * tf.stop_gradient(
                advantage) + entropy_coeff * entropy)

            # Compute the critic loss
            critic_loss = tf.reduce_mean(
                tf.square(tf.stop_gradient(reward + (1 - done) * self.gamma * next_value) - value))

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
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)
        self.__train(state, action, reward, next_state, done)
