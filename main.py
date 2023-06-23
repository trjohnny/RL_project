import argparse
import gymnasium as gym
from agents.on_policy.a2c_agent import A2CAgent
from agents.off_policy.ddpg_agent import DDPGAgent


def main(arguments):
    # create environment
    env = gym.make(arguments.env)

    num_states = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
    num_actions = env.action_space.shape[0]
    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    # Creating agent
    if arguments.algo == 'A2C':
        agent = A2CAgent(num_states, num_actions, lower_bound, upper_bound)
    elif arguments.algo == 'DDPG':
        agent = DDPGAgent(num_states, num_actions, lower_bound, upper_bound)
    else:
        print(f"No algorithm named \"{arguments.algo}\" available")
        return

    agent.train_agent(env, arguments.episodes, arguments.hyperopt)

    # Close the environment
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help='Name of the environment')
    parser.add_argument('--algo', type=str, required=True, help='Algorithm to use', choices=['A2C', 'DDPG'])
    parser.add_argument('--episodes', type=int, required=True, help='Number of episodes')
    parser.add_argument('--hyperopt', action='store_true', help='Use hyperparameter optimization')
    args = parser.parse_args()
    main(args)
