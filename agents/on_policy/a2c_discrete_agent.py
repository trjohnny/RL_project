import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, Model
from agents.agent import Agent


class A2CDiscreteActor(Model):
    def __init__(self, policy_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_model = policy_model

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
        res = tf.convert_to_tensor( self.policy(inputs) )
        res = tf.squeeze(res, axis=1)
        return res


class A2CDiscreteAgent(Agent):
    @staticmethod
    def get_algo():
        return 'A2C_DISCRETE'

    def __init__(self, *agent_params, discrete_values=50, entropy_coeff=1e-2):
        super().__init__(*agent_params)

        self.actor = self.get_actor()
        self.critic = self.get_critic()

        self.entropy_coefficient = entropy_coeff
        self.discrete_values = discrete_values

        # indices
        self.finished = 0
        self.dones = 0

    def get_actor(self):
        input_layer = layers.Input(shape=self.state_shape)
        hidden_layer = layers.Dense(64, activation='relu')(input_layer)
        hidden_layer = layers.Dense(64, activation='relu')(hidden_layer)

        final_layers = []
        for i in range(self.action_shape[0]):
            final_layers.append(layers.Dense(self.discrete_values, activation='softmax')(
                layers.Dense(64, activation='relu')(hidden_layer)))

        policy = Model(inputs=input_layer, outputs=final_layers)

        return A2CDiscreteActor(policy)

    def get_critic(self):
        # State as input
        input_layer = layers.Input(shape=self.state_shape)

        hidden_layer = layers.Dense(64, activation='relu')(input_layer)
        hidden_layer = layers.Dense(64, activation='relu')(hidden_layer)

        output_layer = layers.Dense(self.action_shape[0])(hidden_layer)

        return Model(inputs=input_layer, outputs=output_layer)

    def act(self, state):

        means = self.actor(state)
        actions = []
        # probability = []
        for i in range(len(means)):
            action_probs = means[i].numpy()  # Convert tensor to numpy array
            action = np.random.choice(np.arange(0, len(action_probs), 1), p=action_probs)
            actions.append(action)

        return 2*np.array(actions)/self.discrete_values-1

    @tf.function
    def __train(self, state, action, reward, next_state, done, grad_clip, entropy_coeff=0.00):
        action_index = int(((action+1)*self.discrete_values)/2)
        with tf.GradientTape(persistent=True) as tape:

            pre_means = self.actor(state)  # array of tensors
            # pre_means is a vector of tensors, each tensor is a vector of probabilities

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
