import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt 
import pickle
from matplotlib import style
import time

style.use('ggplot')

#constants

#grid size
SIZE = 20 

#how many episodes
HM_EPISODES = 35_000

MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 3000

EPISODE_STEPS = 800

start_q_table = None # or filename...

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}

class Blob(object):
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
        #does handle things spawning over each other
    
    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):
        if not x:
            #random move
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if not y:
            #random move
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        #limits of grid
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        
        if self.y < 0: 
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1

#observation space is the relative position of the food and the enemy i.e. two tuples
#(x1, y1), (x2, y2)

if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)
        

#training
episode_rewards = []

for episode in range(HM_EPISODES):

    player = Blob()
    food = Blob()
    enemy = Blob()
    
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0

    #steps per episode
    for i in range(EPISODE_STEPS):
        obs = (player - food, player - enemy)

        #explore versus exploit
        if np.random.random() > epsilon:
            #exploit
            action = np.argmax(q_table[obs])
        else:
            #explore
            action = np.random.randint(0, 4)

        player.action(action)

        ## note: could move enemy and food
        ## food.move()
        ## enemy.move()
        #######

        #result of move
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        #remember we have moved
        new_obs = (player - food, player - enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        #update q
        if reward == FOOD_REWARD:
            #end
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY #end enenmy
        else:
            #bellman equation
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[obs][action] = new_q

        #show the environment
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[food.y][food.x] = d[FOOD_N]
            env[player.y][player.x] = d[PLAYER_N]
            env[enemy.y][enemy.x] = d[ENEMY_N]

            img = Image.fromarray(env, "RGB")
            img = img.resize((300, 300))
            cv2.imshow("", np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break
    
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY, )) / SHOW_EVERY, mode='valid')

plt.plot(np.arange(len(moving_avg)), moving_avg)
plt.ylabel(f"reward {SHOW_EVERY} MA")
plt.xlabel("Episode")
plt.show()

with open(f"q_table-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)




