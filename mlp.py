# -*- coding: utf-8 -*-
"""MLP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16MQbaAkb0y_iroh8ck9cVU6yMH5mwEio
"""

#This is a simple example solution of the lab exercise in Week 3 Attack of Flee game. Please use this with the lab session slides and lab exercise brief.
import numpy as np
import matplotlib.pyplot as plt
learningRate = 0.15
max_iter = 600000
mse = 0.18


class neural_network():
    
    def __init__(self):
        #defining the architecture of the network
        input_NUM=4
        hidden_NUM=3
        output_NUM=3
        #randaming the weights (please note we did not use bias in this example solution)
        self.input_hidden_weights = np.random.rand(input_NUM, hidden_NUM)
        self.hidden_output_weights = np.random.rand(hidden_NUM, output_NUM)
        self.error_list = []

    def sigmoid(self, x):
        #defining the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        #defining the derivative (gradient) of the Sigmoid function 
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, epoches, mse): 
        #training the model by updating the weights
        epochs = 0
        mean_squared_error=999999
        while (epochs < max_iter and mean_squared_error > mse):
            o = self.feedforward(training_inputs)
            self.backpropogation(training_inputs,training_outputs,o)
            mean_squared_error = (np.abs(self.error).mean())
            self.error_list.append(mean_squared_error)
            epochs += 1

    def backpropogation(self, training_inputs, training_outputs, output):
            #producing the output
            output = self.feedforward(training_inputs)

            #computing the absolute errors
            self.error = training_outputs - output
            
            #The errors based on the equation in Slide 13 of lab session slides
            self.output_error = self.error * self.sigmoid_derivative(output)
            self.hidden_error = (np.matmul(self.output_error , np.matrix.transpose(self.hidden_output_weights)))*self.sigmoid_derivative(self.hidden) 

            #Weight adjusting based on the equation in Slide 14 of lab session slides
            self.hidden_output_weights += learningRate * np.matmul(np.matrix.transpose(self.hidden), self.output_error)
            self.input_hidden_weights += learningRate * np.matmul(np.matrix.transpose(training_inputs), self.hidden_error)

    def feedforward(self, inputs):  
        inputs = inputs.astype(float)
        #calculating the hidden layer neuron values  
        self.hidden = self.sigmoid(np.matmul(inputs, self.input_hidden_weights))
        #calculating the hidden layer neuron values  
        output = self.sigmoid(np.matmul(self.hidden, self.hidden_output_weights))
        return output

    def error_curve(self):
        #ploting the error curve
        plt.plot(range(len(self.error_list)), self.error_list)
        plt.title('summation of mean squared error')
        plt.xlabel('epoch')
        plt.ylabel('error')      
        



if __name__ == "__main__":

    #initialising the neural_network class
    ann = neural_network()
    
    #printing out the random weights
    print("Randomly generated (intialised) Input layer to hidden layer weights: ")
    print(ann.input_hidden_weights)
    print("Randomly generated (intialised) Hidden layer to output layer weights: ")
    print(ann.hidden_output_weights)

    #the training data inputs
    training_inputs = np.array([[0,		1,		0,	0.2],
                                [0,		1,		1,	0.2],
                                [0,		1,		0,	0.8],
                                [0.1,		0.5,	0,	0.2],
                                [0,		0.25,	1,	0.5],
                                [0,		0.2,	1,	0.2],
                                [0.3,		0.2,	0,	0.2],
                                [0,		0.2,	0,	0.3],
                                [0,		1,		0,	0.2],
                                [0,		1,		1,	0.6],
                                [0,		1,		0,	0.8],
                                [0.1,		0.2,	0,	0.2],
                                [0,		0.25,	1,	0.5],
                                [0,		0.6,	0,	0.2],])

    #the training data outputs
    training_outputs = np.array([[0.9,		0.1,		0.1],
                                 [0.9,		0.1,		0.1],
                                 [0.1,		0.1,		0.1],
                                 [0.9,		0.1,		0.1],
                                 [0.1,		0.9,		0.1],
                                 [0.1,		0.1,		0.9],
                                 [0.9,		0.1,		0.1],
                                 [0.1,		0.9,		0.1],
                                 [0.1,		0.9,		0.1],
                                 [0.1,		0.1,		0.1],
                                 [0.1,		0.9,		0.1],
                                 [0.1,		0.1,		0.9],
                                 [0.1,		0.1,		0.9],
                                 [0.1,		0.1,		0.9],])

    #training the model
    ann.train(training_inputs, training_outputs, max_iter, mse)

    #printing out the trained weights
    print("Trained input layer to hidden layer weights: ")
    print(ann.input_hidden_weights)
    print("Trained hidden layer to output layer weights: ")
    print(ann.hidden_output_weights)

    #test data instance input
    test_inputs = np.array([0,		0.4,	1,	0.4])

    #printing the predicted output
    print("The output of the testing istance is: ")
    print(ann.feedforward(test_inputs))

    #printing the error curve
    ann.error_curve()