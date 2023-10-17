from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, agent):
        self.Agent = agent.__class__
        self.n_states = agent.n_states
        self.n_actions = agent.n_actions
        self.lower_bound = agent.lower_bound
        self.upper_bound = agent.upper_bound

    def objective(self, *hyperparameters, episodes):
        # training the Agent

        agent = self.Agent(self.n_states, self.n_actions, self.lower_bound, self.upper_bound, *hyperparameters)
        rewards1 = agent.train(agent, episodes=episodes)

        agent = self.Agent(self.n_states, self.n_actions, self.lower_bound, self.upper_bound, *hyperparameters)
        rewards2 = agent.train(agent, episodes=episodes)

        agent = self.Agent(self.n_states, self.n_actions, self.lower_bound, self.upper_bound, *hyperparameters)
        rewards3 = agent.train(agent, episodes=episodes)

        # FIN = agent.finished
        TOT_REW = (sum(rewards1) + sum(rewards2) + sum(rewards3)) / (3 * len(rewards1))
        # DONE_RATIO = agent.dones / len(rewards1)

        return -TOT_REW

    @abstractmethod
    def optimize(self, episodes):
        pass
