from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import gym
import numpy as np
env = gym.make("LunarLander-v2")
env.action_space.seed(42)

obs, info = env.reset(seed=42, return_info=True)

def get_state(obs):
    rounded_obs = []
    for o in obs:
        rounded_obs.append(round(o, 2))
    return ((np.array(rounded_obs) + 5) * 10).astype(int)

#Create everything we need for training
#agent = Linear([9])
agent = Sequential()
agent.add(Input(shape=(8)))
agent.add(Dense(9))
agent.add(Dense(9))
agent.add(Dense(4))
agent.load_weights("./LunarLanderModel")
agent.compile(optimizer='adam', loss='mse')
done = False
round = 0
total_rounds=50
X = []
Y = []

def get_action(obs, agent):
    expected_rewards = []
    expected_rewards = agent.predict(np.array([obs]))
    
    a = np.argmax(expected_rewards[0])
    
    return a, expected_rewards[0]

next_act = 0
act = 0
act_hist = []
while round < total_rounds:

    if False:
        act = env.action_space.sample()
        er = "No way hogay we yearnin'"
    else:
        act, er = get_action(obs, agent)

    act_hist.append(act)
    new_obs, reward, done, info = env.step(act)
    #print(f'Round : {round} \nExpected Rewards: {er} Reward: {reward} \n Done: {done} info {info}')
    #print(f'Action: {act}\n')

    env.render()

    #Update X and Y with [obs,action] and reward
    y = []
    for i in range(4):
        if i == act:
            y.append(reward)
        else:
            y.append(0)

    if len(X) == 0:
        X = np.array([obs])
        Y = np.array([y])
    else:    
        X = np.append(X, np.array([obs]), axis=0)
        Y = np.append(Y, np.array([y]), axis=0)
    
    #update previous rewards
    decay = 0.75
    for i, a in enumerate(act_hist):
        Y[i][a] += reward * decay
        decay *= 0.4

    obs = new_obs

    if done:
        obs, info = env.reset(return_info=True)
        round += 1
        #End of the round, do some learning!
        agent.fit(X, Y, epochs=5)

        X = []
        Y = []
        act_hist = []

#Save the model!
agent.save_weights('./LunarLanderModel')

env.close()