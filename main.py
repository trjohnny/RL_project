import argparse
import logging

import gymnasium as gym
from agents.on_policy.a2c_agent import A2CAgent
from agents.off_policy.ddpg_agent import DDPGAgent
from agents.on_policy.a2c_discrete_agent import A2CDiscreteAgent
from agents.on_policy.a2c_n_step_ahead_agent import A2CNStepAheadAgent
import panda_gym
import optimizers.hyperparameters as hyp
from optimizers.differental_evolution import DEA
from optimizers.tree_parzen_estimator import TPE


def main(arguments):
    logging.basicConfig(level=logging.INFO)
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
    elif arguments.algo == 'A2C_DISCRETE':
        agent = A2CDiscreteAgent(num_states, num_actions, lower_bound, upper_bound)
    elif arguments.algo == 'A2C_N_STEP_AHEAD':
        agent = A2CNStepAheadAgent(num_states, num_actions, lower_bound, upper_bound)
    else:
        print(f"No algorithm named \"{arguments.algo}\" available")
        return

    # Use hyperparameter optimization
    if arguments.hyperopt is not None:
        opt = get_optimizer(agent, arguments.hyperopt, arguments.episodes, None)
        # TODO: add hyperopt parameter handling
        result = opt.optimize(arguments.episodes)
        # TODO: Print cool stuff of result
        return

    agent.train_agent(env, arguments.episodes, arguments.verbose)

    # Close the environment
    env.close()


def get_optimizer(agent, hyperopt, *params):
    if hyperopt == 'TPE':
        return TPE(agent, hyp.get_hyp(agent.get_algo()), *params)
    elif hyperopt == 'DEA':
        return DEA(agent, hyp.get_boundaries(agent.get_algo()), *params)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help='Name of the environment')
    parser.add_argument('--algo', type=str, required=True, help='Algorithm to use', choices=['A2C', 'DDPG', 'A2C_N_STEP_AHEAD', 'A2C_DISCRETE'])
    parser.add_argument('-v', '--verbose', type=int, help='Level of verbosity (1, 2, 3)', choices=[1, 2, 3])
    parser.add_argument('--episodes', type=int, required=True, help='Number of episodes')
    parser.add_argument('--hyperopt', type=str, help='Specify hyperparameter optimization algorithm (TPE or DEA)')
    parser.add_argument('--seed', type=int, help='Seed trials for TPE')
    parser.add_argument('--trials', type=int, help='Total trials for TPE > seed')
    parser.add_argument('--gamma', type=int, help='Gamma threshold for TPE')
    args = parser.parse_args()
    main(args)
