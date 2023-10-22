import numpy as np
import tensorflow as tf


class ReplayBuffer:
    def __init__(self, buffer_capacity=10000, batch_size=128):
        self.buffer_capacity = int(buffer_capacity)
        self.batch_size = int(batch_size)
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, 9))
        self.action_buffer = np.zeros((self.buffer_capacity, 3))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, 9))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))

    def add_experience(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]

        self.buffer_counter += 1

    def sample_batch(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        done_batch = tf.convert_to_tensor(self.done_buffer[batch_indices])
        done_batch = tf.cast(done_batch, dtype=tf.float32)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
