from random import random


from activation import Activation
import numpy as np

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons, n_outputs):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        #self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) -- Potential initialisisation
        self.neural_net = list()
        self.learning_rate = 0.1

       
        hidden_layer = [{'weights':[random() for i in range(self.n_inputs + 1)]} for i in range(self.n_neurons)]
        output_layer = [{'weights':[random() for i in range(self.n_neurons + 1)]} for i in range(self.n_outputs)]
        self.neural_net.append(hidden_layer)
        self.neural_net.append(output_layer)
        for layer in self.neural_net:
            print(layer)
        
    @classmethod # Allows to be used without instantiating the class (polymorphism)
    def forward_pass(self, neural_net, row):
        inputs = row
        for layer in neural_net:
            new_inputs = [] # Inputs from each layer
            for neuron in layer: # Taking activation output from each neuron in the layer
                activation = Activation(inputs, neuron['weights']) # Initialising weights and inputs
                neuron['output'] = activation.sigmoid_activation() # Activation function
                new_inputs.append(neuron['output']) # inputs for next layer
            inputs = new_inputs
        return inputs
    
    # Method to update weights after the error backpropagation has been performed
    def update_weights(self, layer):
        for i in range(len(self.neural_net)):
            neuronal_inputs = layer[:-1] # Excluding the output layer?
            if i != 0:
                for neuron in self.neural_net[i-1]:
                    neuronal_inputs = [neuron['output']]
            for neuron in self.neural_net[i]:
                for z in range(len(neuronal_inputs)):
                    neuron['weights'][z] -= np.multiply(self.learning_rate, neuron['error_signal'], neuronal_inputs[z])
                neuron['weights'][-1] -= np.multiply(self.learning_rate, neuron['error_signal'])
            
