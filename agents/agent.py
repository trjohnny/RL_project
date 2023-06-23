from abc import abstractmethod
from abc import ABC
import tensorflow as tf
class Agent(ABC):
    def __init__(self,
                 n_states,
                 n_actions,
                 lower_bound,
                 upper_bound,
                 gamma=.95,
                 actor_learning_rate=1e-3,
                 critic_learning_rate=2e-3,
                 n_layers_actor=2,
                 n_layers_critic=2,
                 units_per_layer_actor=256,
                 units_per_layer_critic=256,
                 activation_actor='relu',
                 activation_critic='relu'):

        self.gamma = gamma
        self.state_shape = (n_states,)
        self.action_shape = (n_actions,)
        self.n_layers_actor = n_layers_actor
        self.n_layers_critic = n_layers_critic
        self.units_per_layer_actor = units_per_layer_actor
        self.units_per_layer_critic = units_per_layer_critic
        self.activation_actor = activation_actor
        self.activation_critic = activation_critic
        self.actor = self.__get_actor()
        self.critic = self.__get_critic()

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_learning_rate)

        # indices
        self.finished = 0
        self.dones = 0

    @abstractmethod
    def __get_actor(self):
        pass

    @abstractmethod
    def __get_critic(self):
        pass

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def train(self, state, action, reward, next_state, done):
        pass
