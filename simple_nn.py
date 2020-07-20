#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

based on: https://towardsdatascience.com/inroduction-to-neural-networks-in-python-7e0b422e6c24
Created on Sun Jul 19 15:45:02 2020

@author: detlef
"""


import numpy as np # helps with the math
import matplotlib.pyplot as plt # to plot error during training


lr = 0.1
hidden_size = 3
lm_w = 1

np.seterr(under='ignore', over='ignore')

# input data
inputs = np.array([[0, 1, 0],
                   [0, 1, 1],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [1, 0, 1]])
# output data
outputs = np.array([[0], [1], [1], [1], [0], [1]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return (np.exp(-x) / (np.exp(-x) + 1) ** 2) #x * (1 - x) # this was an optimized derivative using the self.hidden

# create NeuralNetwork class
class NeuralNetwork:

    # intialize variables in class
    def __init__(self, x, y):
        self.inputs  = x
        self.y = y
        # initialize weights as .50 for simplicity
        self.weights1   = np.random.rand(self.inputs.shape[1], hidden_size) 
        self.weights2   = np.random.rand(hidden_size, 1)                 
        self.error_history = []
        self.epoch_list = []

    # data will flow through the neural network.
    def feed_forward(self):
        self.layer1 = sigmoid(np.dot(self.inputs, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

 
    # going backwards through the network to update weights
    def backpropagation(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        self.error = self.y - self.output
        d_weights2 = np.dot(self.layer1.T, (2*(self.error) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.inputs.T,  (np.dot(2*(self.error) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1 * lr
        self.weights2 += d_weights2 * lr
        
        self.weights1.clip(-lm_w, lm_w)
        self.weights2.clip(-lm_w, lm_w)
        
        #print(d_weights1)
        
    # train the neural net for 25,000 iterations
    def train(self, epochs=200000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()    
            # keep track of the error history over each epoch
            self.error_history.append(np.sum(np.square(self.error)))
            self.epoch_list.append(epoch)
           
    # function to predict output on new and unseen input data                               
    def predict(self, new_input):
        layer1 = sigmoid(np.dot(new_input, self.weights1))
        prediction = sigmoid(np.dot(layer1, self.weights2))
        return prediction

# create neural network   
NN = NeuralNetwork(inputs, outputs)
# train neural network
NN.train()

for i in range(len(inputs)):
    print(NN.predict(inputs[i]), 'correct', outputs[i])

# plot the error over the entire training duration
plt.figure(figsize=(15,5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()

print('Error',NN.error_history[-1])