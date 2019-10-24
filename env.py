import matplotlib.pyplot as plt
import pandas as pd

class env:
	def __init__(self, data, window):
		self.data = data
		self.window = window
		self.episode_total = [] # metric to see total profit of each ep

	def reset(self):
		self.history = [] # to plot buy and sells on closing price graph
		self.inventory = []
		self.profit = 0

	def step(self,action,t):
		
		current_price = self.data[t]
		reward = 0

		# BUY / SELL inventory
		if action == 1: # buy
			self.inventory.append(current_price)
			self.history.append(1)

			print (str(t) + ': BUY')

		elif action == 0 and self.inventory: # sell
			prev_price = self.inventory.pop(0)
			reward = current_price - prev_price

			self.profit += reward
			self.history.append(2)

			print (str(t) + ': SELL, REWARD: ' + str(reward))
		else:
			self.history.append(0)

		done = True if t == len(self.data) - self.window - 1 else False

		if done:
			self.episode_total.append(self.profit)

		return reward, done

	def plot_episodes(self,file_name):
		fig = plt.figure(len(self.episode_total)+1)
		plt.plot( range(len(self.episode_total)), self.episode_total)

		plt.xlabel('Episode')
		plt.ylabel('Reward Total')
		plt.title(file_name+': Total Reward per Episode')

		plt.savefig('images/'+file_name+'/Total_Rewards'+'.png')

	def plot_history(self,episode,dates,file_name):
		df = pd.DataFrame({'Dates':dates[:-self.window],
			'Data':self.data[:-self.window],
			'Action':self.history})

		buy = df[df['Action']==1]
		sell = df[df['Action']==2]

		plt.figure(episode)
		plt.scatter(buy['Dates'],buy['Data'],
			color='g',s=10,zorder=3,label='Buy')
		plt.scatter(sell['Dates'],sell['Data'],
			color='r',s=10,zorder=2,label='Sell')
		plt.plot(dates,self.data,zorder=1)

		plt.gcf().autofmt_xdate()

		plt.legend(title='Action')
		plt.xlabel('Time')
		plt.ylabel('Price')
		plt.title('Episode ' + str(episode))

		plt.savefig('images/'+file_name+'/episode_'+str(episode)+'.png')