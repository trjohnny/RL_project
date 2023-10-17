import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.python.keras import layers, Model
from tensorflow.python.keras.initializers import Constant, Zeros
from agents.agent import Agent

"""
This class is a placeholder for a future real implementation, I don't think this can actually work...
"""
class A2CActor(Model):
    def __init__(self, policy_model, independent_log_stds=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_model = policy_model
        self.independent_log_stds = independent_log_stds

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
        if self.independent_log_stds is None:
            means, std_devs = self.policy(inputs)
            std_devs = tf.clip_by_value(tf.math.exp(std_devs), 1e-3, 50)
        else:
            means = self.policy(inputs)
            std_devs = [tf.clip_by_value(tf.math.exp(logstd), 1e-3, 50) for logstd in self.independent_log_stds]
            std_devs = tf.stack(std_devs)
        return means, std_devs


class A2CNGAgent(Agent):
    @staticmethod
    def get_algo():
        return 'A2C'

    def __init__(self, *agent_params, std_state_dependent=True, log_std_init=.5, entropy_coeff=1e-2):
        super().__init__(*agent_params)

        self.std_state_dependent = std_state_dependent
        self.log_std_init = log_std_init

        self.actor = self.get_actor()
        self.critic = self.get_critic()

        self.entropy_coefficient = entropy_coeff

        # indices
        self.finished = 0
        self.dones = 0

    def get_actor(self):
        input_layer = layers.Input(shape=self.state_shape)
        hidden = input_layer

        for i in range(int(self.n_layers_actor) if not self.std_state_dependent else int(self.n_layers_actor) - 1):
            hidden = layers.Dense(int(self.units_per_layer_actor), activation=self.activation_actor)(hidden)

        if self.std_state_dependent:
            hidden_m = layers.Dense(int(self.units_per_layer_actor), activation=self.activation_actor)(hidden)
        else:
            hidden_m = hidden

        means = layers.Dense(self.action_shape[0], activation='tanh')(hidden_m)
        independent_logstds = None

        if self.std_state_dependent:
            hidden_s = layers.Dense(int(self.units_per_layer_actor), activation=self.activation_actor)(hidden)
            dependent_logstds = layers.Dense(self.action_shape[0],
                                             kernel_initializer=Zeros(),
                                             bias_initializer=Constant(self.log_std_init))(hidden_s)
            policy = Model(inputs=input_layer, outputs=[means, dependent_logstds])

        else:
            independent_logstds = []
            for i in range(self.action_shape[0]):
                independent_logstds.append(Model.add_weight(name=f'logstd_{i}',
                                                            shape=(),
                                                            initializer=Constant(self.log_std_init),
                                                            trainable=True))
            policy = Model(inputs=input_layer, outputs=means)

        return A2CActor(policy, independent_logstds)

    def get_critic(self):
        # State as input
        input_layer = layers.Input(shape=self.state_shape)
        hidden = input_layer

        for i in range(int(self.n_layers_critic)):
            hidden = layers.Dense(int(self.units_per_layer_critic), activation=self.activation_critic)(hidden)

        outputs = layers.Dense(1)(hidden)

        # Outputs single value for give state-action
        model = tf.keras.Model(input_layer, outputs)

        return model

    def act(self, state):
        means, std_devs = self.actor(state)
        action = np.random.normal(loc=means, scale=std_devs)
        return action.flatten()

    @tf.function
    def __train(self, state, action, reward, next_state, done, grad_clip, entropy_coeff=0.00):
        with tf.GradientTape(persistent=True) as tape:
            means, std_devs = self.actor(state)
            value = self.critic(state)

            next_value = self.critic(next_state)

            action_prob = tf.exp(-0.5 * tf.square((action - means) / std_devs)) / (std_devs * tf.sqrt(2. * np.pi))
            action_prob = tf.reduce_prod(action_prob, axis=1, keepdims=True)

            advantage = reward + (1 - done) * self.gamma * next_value - value

            # Compute the entropy of the current policy
            entropy = 0.5 * (tf.math.log(2 * np.pi * std_devs ** 2) + 1)
            entropy = tf.reduce_sum(entropy, keepdims=True)

            # Compute the actor loss
            actor_loss = -tf.reduce_mean(
                tf.math.log(tf.maximum(action_prob, 5e-6)) * tf.stop_gradient(advantage) + entropy_coeff * entropy)

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
