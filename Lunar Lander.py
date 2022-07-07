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
total_rounds=100
X = []
Y = np.array([])

agent = Linear([9])
new_obs = []

while round < total_rounds:
    
    if 0<round:
        #Do some learning
        a = observation.tolist()
        a.append(act)
        X.append(a)
        Y = np.append(Y, np.array([reward]))
        agent.fit(X=np.array(X), Y=Y, epochs=10)
        observation = new_obs
    
    act = env.action_space.sample()
    #Get next action by taking the least most expected reward
    to_pred = [observation.copy() for i in range(4)]
    for i in range(4):
        np.append(to_pred[i], i)
        to_pred[i] = np.array(to_pred[i])
    to_pred = np.array(to_pred) #Yeah, I know this is ugly Numpy array conversions...
    reward_preds = agent.predict(to_pred)

    max_reward = reward_preds[0]
    for i in range(len(reward_preds[0:])):
        if reward_preds[i] > max_reward:
            max_reward = reward_preds[i]
            act = i

    new_obs, reward, done, info = env.step(act)
    print(f' Reward: {reward} \n Done: {done} info {info}')
    print(f'Expected Reward: {max_reward}')
    print(f'Action: {act}')
    env.render()

    if done:
        observation, info = env.reset(return_info=True)
        round += 1
        X = []

env.close()