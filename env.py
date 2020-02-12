import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class env:
	def __init__(self, data, closing_data, window):
		self.data = data
		self.closing_data = closing_data
		self.window = window
		self.episode_total = [] # metric to see total profit of each ep

	def reset(self):
		self.history = [] # to plot buy and sells on closing price graph
						  # 0 = HOLD, 1 = BUY, 2 = SELL
		self.inventory = []
		self.profit = 0 # current profit of the episode (normalised)
		self.profit_history = []

	def step(self,action,t):
		
		current_price = self.data[t+self.window]
		reward = 0

		if action == 1: # BUY
			self.inventory.append(current_price)
			self.history.append(1)

			print (str(t) + ': BUY ' + str(self.unnormalise(current_price)))

		elif action == 0 and self.inventory: # SELL
			prev_price = self.inventory.pop(0)
		
			if current_price > prev_price: # reward of only positive profits
				reward = current_price - prev_price

			#self.profit += current_price - prev_price
			self.profit += self.unnormalise(current_price) - self.unnormalise(prev_price)

			self.history.append(2)

			print (str(t) + ': SELL '+str(self.unnormalise(current_price))+', REWARD: ' + str(reward))

		else: # HOLD
			self.history.append(0)

		self.profit_history.append(self.profit)

		done = True if t == len(self.data) - self.window - 1 else False

		if done:
			self.episode_total.append(self.profit)

		return reward, done

	def unnormalise(self,normalised_data):
		return (normalised_data * (self.closing_data.max() - self.closing_data.min()) ) + self.closing_data.min()

	def plot_history(self,episode,dates,file_name):

		un_data = self.unnormalise(self.data)

		df = pd.DataFrame({'Dates':dates[self.window:],
			'Data':un_data[self.window:],
			'Action':self.history})

		buy = df[df['Action']==1]
		sell = df[df['Action']==2]

		fig, ax1 = plt.subplots()

		c1 = 'tab:blue'
		ax1.set_xlabel('Time')
		ax1.set_ylabel('Share Price')
		ax1.tick_params(axis='y')

		ax1.scatter(buy['Dates'],buy['Data'],
			color='g',s=4,zorder=3,label='Buy')
		ax1.scatter(sell['Dates'],sell['Data'],
			color='r',s=4,zorder=2,label='Sell')
		ax1.plot(dates,un_data,color=c1,zorder=1,label='Closing Price')

		ax2 = ax1.twinx()

		c2 = 'tab:orange'
		ax2.tick_params(axis='y')
		ax2.set_ylabel('Profit')
		ax2.plot(dates[self.window:],self.profit_history,
			color=c2,zorder=0,label='Profit')

		h1, l1 = ax1.get_legend_handles_labels()
		h2, l2 = ax2.get_legend_handles_labels()
		fig.legend(h2+h1,l2+l1,loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax1.transAxes)

		plt.title('Episode ' + str(episode) + ', Total Rewards: ' + 
			"{0:.2f}".format( self.profit) )
		fig.tight_layout()
		plt.gcf().autofmt_xdate()

		plt.savefig('images/'+file_name+'/episode_'+str(episode)+'.png')

	# PLOT GRAPHS PER EPISODE
	def plot_episodes(self,file_name):
		fig = plt.figure(len(self.episode_total)+1)
		plt.plot( range(len(self.episode_total)), self.episode_total)

		plt.xlabel('Episode')
		plt.ylabel('Reward Total')
		plt.title(file_name+': Total Reward per Episode')

		plt.savefig('images/'+file_name+'/Total_Rewards'+'.png')

	
