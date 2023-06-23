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

    @staticmethod
    def __run_episode(env):
        state, info = env.reset()

        state = np.concatenate([state['observation'][:6], state['desired_goal']], dtype=np.float32)

        state_tensor = tf.convert_to_tensor([state], dtype='float32')

        total_reward = 0
        done = False
        truncated = False
        num_step = 0

        while not done and not truncated:
            action = self.act(state_tensor)

            next_state, reward, done, truncated, info = env.step(action)
            reward *= 100

            if done and num_step <= 1:
                return None

            if env.unwrapped.spec.id == "PandaPushDense-v3":
                reward -= 0.5 * np.linalg.norm(next_state['observation'][:3] - next_state['achieved_goal'])
                reward -= 0.1 * np.linalg.norm(action) ** 2

            next_state = np.concatenate([next_state['observation'][:6], next_state['desired_goal']], dtype=np.float32)

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

    @staticmethod
    def train_agent(env, episodes, hyperopt=False, verbose=0):
        rewards = []
        mod = episodes - 1
        if verbose == 1:
            mod = 100
        elif verbose == 2:
            mod = 10
        elif verbose == 3:
            mod = 1

        # Use hyperparameter optimization
        if hyperopt:
            optimizer = TPE(self.__class__, hyp.get_hyp("placeholder"))
            trials, EIs = optimizer.fmin()
            return

        for episode in range(1, episodes + 1):

            reward = self.__run_episode(env)

            if reward is not None:
                rewards.append(reward)
            else:
                continue

            # Print the reward for each episode
            if episode % mod == 0:
                print(f'Episode {episode}: Reward: {reward:.2f}')

        return rewards
