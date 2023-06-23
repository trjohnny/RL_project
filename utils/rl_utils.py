import numpy as np
import tensorflow as tf

from optimizers.tpe import TPE


def __run_episode(env_, agent, episode):
    state, info = env_.reset()

    state = np.concatenate([state['observation'][:6], state['desired_goal']], dtype=np.float32)

    state_tensor = tf.convert_to_tensor([state], dtype='float32')

    total_reward = 0
    done = False
    truncated = False
    num_step = 0

    while not done and not truncated:
        action = agent.act(state_tensor, agent.noise, num_step, episode)

        next_state, reward, done, truncated, info = env_.step(action)
        reward *= 100

        if done and num_step <= 1:
            return None

        if env_.unwrapped.spec.id == "PandaPushDense-v3":
            reward -= 0.5 * np.linalg.norm(next_state['observation'][:3] - next_state['achieved_goal'])
            reward -= 0.1 * np.linalg.norm(action) ** 2

        next_state = np.concatenate([next_state['observation'][:6], next_state['desired_goal']], dtype=np.float32)

        agent.train(state, action, reward, next_state, done)

        state = next_state
        state_tensor = tf.convert_to_tensor([state], dtype='float32')
        total_reward += reward
        num_step += 1

    if not done:
        agent.finished = episode
    else:
        agent.dones += 1

    return total_reward


def train_agent(env, agent, episodes, hyperopt=False, verbose=0):
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
        optimizer = TPE(agent.__class__, hyp.get_hyp(arguments.algo))
        trials, EIs = optimizer.fmin()
        return

    for episode in range(1, episodes + 1):

        reward = __run_episode(env, agent, episode)

        if reward is not None:
            rewards.append(reward)
        else:
            continue

        # Print the reward for each episode
        if episode % mod == 0:
            print(f'Episode {episode}: Reward: {reward:.2f}')

    return rewards
