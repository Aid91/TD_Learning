import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

# Algorithm parameters
learning_rate = 0.5
gamma = 1.
epsilon = .01
render = False
N_trial = 1000
N_trial_test = 100
trial_duration = 100

# Generate the environment
env = FrozenLakeEnv(map_name='4x4', is_slippery=False)
n_state = env.observation_space.n
n_action = env.action_space.n

# Initialize the Q values
Q_table = np.zeros((n_state, n_action))


def policy(Q_table, state, epsilon):
    '''
       Implementation of the epsilon greedy policy.

       :param Q_table: Table containing the expected return for each action and state pair
       :param state:
       :return:
       '''

    if rd.rand(1) < epsilon:
        action = rd.randint(low=0, high=n_action)
    else:
        max_actions = np.argwhere(Q_table[state] == np.amax(Q_table[state])).flatten()
        action = rd.choice(max_actions)

    return action


def update_Q_table(Q_table, state, action, reward, new_state, new_action, is_done):
    '''
    Update the Q values according to the SARSA algorithm.

    :param Q_table: Table containing the expected return for each action and state pair
    :param state:
    :param action:
    :param reward:
    :param new_state:
    :param new_action:
    :param is_done:
    :return:
    '''

    if is_done:
        delta = (reward - Q_table[state, action])
    else:
        delta = (reward + gamma * Q_table[new_state, new_action] - Q_table[state, action])

    Q_table[state, action] += learning_rate * delta


reward_list = []
for k in range(N_trial + N_trial_test):

    acc_reward = 0  # Init the accumulated reward
    observation = env.reset()  # Init the state
    action = policy(Q_table, observation, epsilon)  # Init the first action

    for t in range(trial_duration):
        if render: env.render()

        new_observation, reward, done, info = env.step(action)  # Take the action
        new_action = policy(Q_table, new_observation, epsilon)
        update_Q_table(Q_table=Q_table, state=observation, action=action, reward=reward, new_state=new_observation,
                       new_action=new_action, is_done=done)

        observation = new_observation  # Pass the new state to the next step
        action = new_action  # Pass the new action to the next step
        acc_reward += reward  # Accumulate the reward
        if done:
            break  # Stop the trial when you fall in a hole or when you find the goal

    reward_list.append(acc_reward)  # Store the result

print('Average accumulated reward in {} test runs: {:.3g}'.format(N_trial_test, np.mean(reward_list[N_trial:])))
