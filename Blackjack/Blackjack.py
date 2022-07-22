#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Input
import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('Blackjack-v1', natural=True, sab=True)
env.action_space.seed(42)
obs, info = env.reset(seed=42, return_info=True) 

Q = np.zeros([32,11,2, 2])

def get_q_action(obs):
    return np.argmax(Q[obs[0]][obs[1]][int(obs[2])])

total_reward = 0 
act_hist = []
round = 0
max_rounds = 1000000
#avg_reward = -0.282342 for 1000000 rounds
done= False

while round < max_rounds:
    act = 0

    if round < max_rounds/2:
        act = env.action_space.sample()
    else:
        act = get_q_action(obs)

    state_act = Q[obs[0]][obs[1]][int(obs[2])][act]
    act_hist.append(state_act)
    obs, reward, done, info = env.step(act)
    #print(f'Action | {act}, Reward | {reward}')
    #update Q with reward
    Q[obs[0]][obs[1]][int(obs[2])][act] += reward

    #env.render()

    if done:
        total_reward += reward
        round += 1
        obs = env.reset()

print(f'avg_reward = {total_reward / max_rounds}')

env.close()