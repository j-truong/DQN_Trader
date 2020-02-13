# DQN_Trader
Developed an algorithmic trading agent implementing deep reinforcement learning (DQN) to simulate trading Google stock shares (daily dataset of 10 years). The following trading agent inputs closing price data in window set intervals to create a trading strategy by optimising the agent's policy by applying the Bellman's Equation under a fixed reward system.
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
With my first model utilising a simple Neural Network to classify the actions of the agent, the trading agent was able to effectively simulate trade with majority obtaining positive profit. The intital episode obtained a profit of £611.14 and the best episode (episode 26) with a profit of £1752.58.

![image](https://github.com/j-truong/DQN_Trader/blob/master/images/GOOGL_10/Model1/episode_0.png)

![image](https://github.com/j-truong/DQN_Trader/blob/master/images/GOOGL_10/Model1/episode_26.png)

![image](https://github.com/j-truong/DQN_Trader/blob/master/images/GOOGL_10/Model1/Total_Rewards.png)

If this model was realistically applied, a model with no initial budget, iterating with one episode, and considering the cost of trade transcations would not result in a great profit over the duration of 10 years.

I wanted to experiement further and implement a different model to compare overall profits to my first model. Below displays graphs implementing a simple LSTM Neural Network to which I believe would be better suited to classfiy actions by analysing the patterns within the closing data. 

The first episode surpringly obtained the largest profit as compared to the rest of the episodes, I am not sure if this is due to the luck of the intitial exploratory section of the model or if the continuation of the model was not suitable when reused onto the same dataset. The total profit for the episode 0 was £4350.03, not only this shows a significant improvement from the first model, but all the episodes seemed to perform better.

![image](https://github.com/j-truong/DQN_Trader/blob/master/images/GOOGL_10/Model2_LSTM/episode_0.png)

![image](https://github.com/j-truong/DQN_Trader/blob/master/images/GOOGL_10/Model2_LSTM/episode_4.png)

![image](https://github.com/j-truong/DQN_Trader/blob/master/images/GOOGL_10/Model2_LSTM/Total_Rewards.png)

In a future model, I would like to input technical indicators as features to feed into the LSTM model. Examples of potenial features would be RSI, volume, and moving averages. I know that theses are popular indicators to stock traders so it may become valuable to the model if it is able to analyse and detect patterns alike stock traders to improve on performance and enhance trading strategy.
