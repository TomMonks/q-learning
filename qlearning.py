import gym
import numpy as np

env = gym.make("MountainCar-v0")



LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25_000
SHOW_EVERY = 2000

epsilon = 0.1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

DISCRETE_OBS_SIZE = [20] * len(env.observation_space.high)

discrete_obs_win_size = (env.observation_space.high - env.observation_space.low) \
    / DISCRETE_OBS_SIZE

print(discrete_obs_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SIZE + [env.action_space.n]))
print(q_table.shape)

def get_discrete_state(state, env):
    discrete_state = (state - env.observation_space.low) / discrete_obs_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    
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
        episode -= epsilon_decay_value
    

env.close()