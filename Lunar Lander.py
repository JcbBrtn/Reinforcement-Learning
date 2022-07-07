# import sys
# sys.path.append('/home/jacob/.local/lib/python3.8/site-packages')
from Linear import Linear
import gym
import numpy as np
env = gym.make("LunarLander-v2")
env.action_space.seed(42)

observation, info = env.reset(seed=42, return_info=True)

done = False
round = 0
total_rounds=1

new_obs = []

while round < total_rounds:
    
    act = env.action_space.sample()
    #Get next action by taking the action with the most expected reward

    new_obs, reward, done, info = env.step(act)
    print(f' Reward: {reward} \n Done: {done} info {info}')
    print(f'Action: {act}')
    env.render()

    if done:
        observation, info = env.reset(return_info=True)
        round += 1
        #End of the round, do some learning!

env.close()