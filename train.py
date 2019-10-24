import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from dqn_agent import dqn_agent
from env import env

window = 10
episodes = 50
batch_size = 32

epsilon = 1 
epsilon_decay = 0.995
min_epsilon = 0.001

file_name = 'GOOGL_10'

df = pd.read_csv('data/'+file_name+'.csv')
data = df['Close']
dates = pd.to_datetime(df['Date'],format="%Y/%m/%d")

agent = dqn_agent(window)
env = env(data,window)

for episode in range(episodes):

	current_state = np.array([data[0:window]])
	done = False

	env.reset()

	for t in range(len(data) - window):

		action = agent.act(current_state, epsilon)

		reward, done = env.step(action,t)
		new_state = np.array([data[t: t + window]])

		agent.memory.append([current_state, action, reward, new_state, done])

		agent.train(batch_size)

		if epsilon > min_epsilon:
			epsilon *= epsilon_decay
			epsilon = max(min_epsilon, epsilon)

		current_state = new_state

		if done:
			print ('---------------------------------------')
			print ('Episode ' + str(episode) + ' Total Rewards: ' + str(env.episode_total[-1]))
			print ('Inventory: ' + str(env.inventory))
			print ('---------------------------------------')

	env.plot_history(episode,dates,file_name)
env.plot_episodes(file_name)

plt.show()
