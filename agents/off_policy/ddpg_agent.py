from agents.agent import Agent
import numpy as np
import tensorflow as tf
from agents.off_policy.replay_buffer import ReplayBuffer
from tensorflow import keras


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.x_prev = np.zeros_like(mean)
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):

        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )

        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            pass


class DDPGAgent(Agent):
    def get_algo(self):
        pass

    def __init__(self, *agent_params, tau=0.005, buffer_size=1_000_000, batch_size=128,
                 start_training=1_000, noise_std=.2):

        super().__init__(*agent_params)

        self.noise = OUActionNoise(mean=np.zeros(self.action_shape[0]),
                                   std_deviation=float(noise_std) * np.ones(self.action_shape[0]))
        self.tau = tf.convert_to_tensor(tau, dtype=tf.float32)

        self.actor = self.get_actor()
        self.critic = self.get_critic()

        self.target_actor = tf.keras.models.clone_model(self.actor)
        self.target_critic = tf.keras.models.clone_model(self.critic)

        self.target_critic.set_weights(self.critic.get_weights())

        self.replay_buffer = ReplayBuffer(self.state_shape[0], buffer_size, batch_size)

        self.start_training = start_training

        # indices
        self.finished = 0
        self.dones = 0

    def get_actor(self):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = keras.layers.Input(shape=self.state_shape)
        hidden = inputs

        for i in range(int(self.n_layers_actor)):
            hidden = keras.layers.Dense(int(self.units_per_layer_actor), activation=self.activation_actor)(hidden)

        outputs = keras.layers.Dense(self.action_shape[0], activation="tanh", kernel_initializer=last_init)(hidden)

        model = keras.Model(inputs, outputs)

        return model

    def get_critic(self):
        # State as input
        state_input = keras.layers.Input(shape=self.state_shape)
        state_out = keras.layers.Dense(16, activation="relu")(state_input)

        # Action as input
        action_input = keras.layers.Input(shape=self.action_shape)
        action_out = keras.layers.Dense(16, activation="relu")(action_input)

        # Both are passed through separate layer before concatenating
        concat = keras.layers.Concatenate()([state_out, action_out])
        hidden = concat

        for i in range(int(self.n_layers_critic)):
            hidden = keras.layers.Dense(int(self.units_per_layer_critic), activation=self.activation_critic)(hidden)

        outputs = keras.layers.Dense(1)(hidden)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def __update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def act(self, state):
        sampled_actions = tf.squeeze(self.actor(state))

        # Adding noise to action
        sampled_actions_noise = sampled_actions.numpy() + self.noise()

        # Make sure action is within bounds
        legal_action = np.clip(sampled_actions_noise, self.lower_bound, self.upper_bound)

        return np.squeeze(legal_action)

    def __train(self, state, action, reward, next_state, done):

        with tf.GradientTape() as tape:
            # Select the next action
            next_action = self.target_actor(next_state, training=True)

            # Compute the target Q value
            target_q = reward + (1 - done) * self.gamma * self.target_critic([next_state, next_action], training=True)

            # Compute the current Q value
            current_q = self.critic([state, action], training=True)

            # Compute the critic loss
            critic_loss = tf.math.reduce_mean(tf.square(target_q - current_q))

        # Compute and apply the gradients for the critic loss
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            # Select the current action
            action = self.actor(state, training=True)

            # Compute the actor loss
            actor_loss = -tf.math.reduce_mean(self.critic([state, action], training=True))

        # Compute and apply the gradients for the actor loss
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

    def train(self, state, action, reward, next_state, done):
        self.replay_buffer.add_experience((state, action, reward, next_state, done))

        if self.replay_buffer.buffer_counter > self.start_training:
            # Sample a batch of experiences from the replay buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch()

            # Train the agent using the batch of experiences
            self.__train(states, actions, rewards, next_states, dones)
            self.__update_target(self.target_actor.variables, self.actor.variables, self.tau)
            self.__update_target(self.target_critic.variables, self.critic.variables, self.tau)
