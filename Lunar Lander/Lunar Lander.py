from json import load
from tabnanny import verbose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, BatchNormalization
from tensorflow.keras.models import load_model
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import gym
import numpy as np
env = gym.make("LunarLander-v2")
env.action_space.seed(42)

CHECKPOINT_PATH=".LunarLanderModel.h5"

obs, info = env.reset(seed=42, return_info=True)

def normalize(obs):
    ob = obs + [1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0 ]
    sc = [3,3,10,10,6.28,10,2,2]
    for i in range(len(ob)):
        ob[i] = ob[i]/sc[i]
    return ob


NEW_MODEL = False
agent = Sequential()
if NEW_MODEL:
    #Create everything we need for training
    #agent = Linear([9])
    agent.add(LSTM(132, input_shape=(8, 1)))
    agent.add(BatchNormalization())
    agent.add(Dense(4))
    #agent.load_weights(CHECKPOINT_PATH)
    agent.compile(optimizer='adam', loss='mse')
else:
    agent = load_model(CHECKPOINT_PATH)
done = False
round = 0
total_rounds=30
X = []
Y = []
X_Hist = []
Y_Hist = []

def get_action(obs, agent):
    expected_rewards = []
    expected_rewards = agent.predict(np.array([obs]), verbose=0)
    
    a = np.argmax(expected_rewards[0])
    
    return a, expected_rewards[0]

next_act = 0
act = 0
act_hist = []
while round < total_rounds:

    obs = normalize(obs)

    if len(X) == 0:
        act, er = get_action(obs, agent)
    else:
        act, er = get_action(X, agent)

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
    decay = 0.85
    for i, a in enumerate(act_hist):
        reward *= decay
        Y[i][a] += reward

    obs = new_obs

    if done:
        obs, info = env.reset(return_info=True)
        round += 1
        print(f'Round {round} / {total_rounds}')

        #Add the new Data to the Master List
        if len(X_Hist) == 0:
            X_Hist = X
            Y_Hist = Y
        else:
            X_Hist = np.append(X_Hist, X, axis=0)
            Y_Hist = np.append(Y_Hist, Y, axis=0)

        #End of the round, do some learning!
        agent.fit(X_Hist, Y_Hist, epochs=10, verbose=0)

        X = []
        Y = []
        act_hist = []

#Save the model!
agent.save(CHECKPOINT_PATH)

env.close()