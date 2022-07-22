from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import gym
import numpy as np
env = gym.make("InvertedPendulum-v2")
env.action_space.seed(42)
obs, info = env.reset(seed=42, return_info=True)

round = 0
max_rounds = 10
done= False

while round < max_rounds:

    act = env.action_space.sample()

    obs, reward, done, info = env.step(act)

    env.render()

    if done:
        round += 1

env.close()
