import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")



LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 10_000
SHOW_EVERY = 500

epsilon = 0.2
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

DISCRETE_OBS_SIZE = [20] * len(env.observation_space.high)

discrete_obs_win_size = (env.observation_space.high - env.observation_space.low) \
    / DISCRETE_OBS_SIZE

print(discrete_obs_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SIZE + [env.action_space.n]))
print(q_table.shape)

ep_rewards = []

agg_ep_rewards = {'ep':[], 'avg':[], 'min':[], 'max':[]}

def get_discrete_state(state, env):
    discrete_state = (state - env.observation_space.low) / discrete_obs_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    #cumulative reward per episode
    episode_reward = 0

    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset(), env)


    done = False
    while not done:
        #explore versus exploit
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(low=0, high=env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        #track the cumulative reward for the episode
        episode_reward += reward

        new_discrete_state = get_discrete_state(new_state, env)
        if render:
            env.render()
        if not done:              
            #use bellman equation to update   
            
            #get maximum future q value for the new state
            max_future_q = np.max(q_table[new_discrete_state])
            
            #current q value for action
            current_q = q_table[discrete_state + (action, )]

            #back propogate the q value using the discount factor
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            #update the q value of the discrete_state, action
            q_table[discrete_state + (action, )] = new_q


        elif new_state[0] >= env.goal_position:
            #if goal has been reached
            print("made it on episode {}".format(episode))
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING > episode > START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        agg_ep_rewards['ep'].append(episode)
        agg_ep_rewards['avg'].append(average_reward)
        agg_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        agg_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print("ep: {0}, avg {1}, min {2}, max {3}".format(episode, 
                                                            average_reward,
                                                            min(ep_rewards[-SHOW_EVERY:]),
                                                            max(ep_rewards[-SHOW_EVERY:])
                                                            ))


plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['avg'], label='Average Reward')
plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['min'], label='Min Reward')
plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['max'], label='Max Reward')
plt.legend(loc=1)
plt.show()
