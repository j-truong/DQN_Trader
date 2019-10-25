# DQN_Trader
Developed an algorithmic trading agent implementing deep reinforcement learning (DQN) to buy and sell stock shares. The following trading agent inputs raw closing price data in set intervals to forecast stock prices then optimises the expected rewards utilising the Bellman's Equation under a fixed policy.
## Prerequisites

```
pip install requirements.txt
```

## Script
Train model
```
python train.py
```

## Results
Using the Google 10 year daily stock market dataset, the agent was able to effectively trade the majority of the episodes, with the best episode recieving an overall profit total of over $600,000 in the course of the 10 years (episode 13), but also resulting in a profit total of $0 in some. 

![image](https://github.com/j-truong/DQN_Trader/blob/master/images/GOOGL_10/Total_Rewards.png)

The trading agent was able to trade more effectively within the lower quartile of the episodes conducted, predicting and detecting when the market will alter its orientation by classifying actions using price patterns fed into the Neural Network. The agent eventually trained itself to buy early and sell towards the end of the dataset within the later episodes (a good example would be episode 48); although it is optimal, it is not idealistic as a trading agent as it is not effectively trading and would be problematic with live or new dissimilar data. 

![image](https://github.com/j-truong/DQN_Trader/blob/master/images/GOOGL_10/episode_0.png)

![image](https://github.com/j-truong/DQN_Trader/blob/master/images/GOOGL_10/episode_13.png)

![image](https://github.com/j-truong/DQN_Trader/blob/master/images/GOOGL_10/episode_48.png)
