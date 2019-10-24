import keras
from keras.models import Sequential
from keras.layers import Dense, CuDNNLSTM, BatchNormalization, Dropout, LSTM
from keras.optimizers import Adam

import numpy as np
from collections import deque
import random
import math

min_replay_memory = 40
replay_memory_size = 50

class dqn_agent:
	def __init__(self,window):
		self.window = window
		self.action_size = 2
		self.state_size = 1
		self.gamma = 0.95
		self.memory = deque(maxlen=replay_memory_size)

		self.model = self.create_model()

	def create_model(self):
		model = Sequential([

			Dense(64, input_shape=(self.window,), activation='relu'),
			Dropout(0.2),

			Dense(32, activation='relu'),
			Dropout(0.2),

			Dense(32, activation='relu'),
			Dropout(0.2),

			Dense(self.action_size, activation='linear')
			])
		model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics = ['accuracy'])
		return model

	def act(self,current_state,epsilon):
		if np.random.random() < epsilon:
			action = np.random.randint(0,self.action_size)
		else:
			action_list = self.model.predict(current_state)
			action = np.argmax(action_list[0])
		return action

	def normalise(self,data):
		normalised_data = (data - data.min()) / (data.max() - data.min())
		return np.array([normalised_data])

	def train(self,batch_size):

		if (len(self.memory) < min_replay_memory):
			return

		minibatch = random.sample(self.memory, batch_size)

		for (current_state, action, reward, new_state, done) in minibatch:
			
			if not done:
				target = reward + self.gamma * np.max(self.model.predict(new_state)[0])
			else:
				target = reward

			target_f = self.model.predict(current_state)
			target_f[0][action] = target
			self.model.fit(current_state, target_f, epochs=1, verbose=0)

