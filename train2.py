# -*- coding: utf-8 -*-
"""
Created on Sat May 20 01:06:13 2023

@author: J Kalyan
"""

# Importing Library's
import numpy as np
from environment import Environment

# Defining the perameters 
gamma = 0.9
alpha = 0.75
n_epochs = 10000

# Environment and Q-Table initialization
env = Environment()
rewards = env.rewardBoard
Qtable = rewards.copy()

# Preparing the Q-Learning process 1
possible_states = list()
for i in range(rewards.shape[0]):
    if sum(abs(rewards[i])) !=0:
        possible_states.append(i)
        
# Preparing the Q-Learning Process 2
def maximum(qvalues):
    inx = 0
    maxQ_value = -np.inf
    for i in range(len(qvalues)):
        if qvalues[i]>maxQ_value and qvalues[i] !=0:
            maxQ_value = qvalues[i]
            inx=i
    return inx,maxQ_value

# Starting the Q-Leaning Process
for epoch in range(n_epochs):
    print('\rEpoch: '+str(epoch+1),end='')
    startingPos = np.random.choice(possible_states)
    
    # Getting all the playable actions
    possibleActions = list()
    for i in range(rewards.shape[1]):
        if rewards[startingPos][i]!=0:
            possibleActions.append(i)
    #Playing a random action
    action = np.random.choice(possibleActions)
    
    reward = rewards[startingPos][action]
    
    #Updating Q-Values
    _, maxQvalue = maximum(Qtable[action])
    
    TD = reward + gamma * maxQvalue - Qtable[startingPos][action]
    Qtable[startingPos][action] = Qtable[startingPos][action]+alpha * TD


# Display the results
currentPos = env.startingPos 
while True:
    action,_ = maximum(Qtable[currentPos])
    env.movePlayer(action)
    
    currentPos = action 
    
        
        
        
        
        

