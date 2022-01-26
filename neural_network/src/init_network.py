from random import random
from activation import Activation

class init_network:

    def __init__(self):
        pass

    def __init__(self, n_inputs, n_neurons, n_outputs):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        neural_net = list()

        
        hidden_layer = [{'weights':[random() for i in range(self.n_inputs + 1)]} for i in range(self.n_neurons)]
        output_layer = [{'weights':[random() for i in range(self.n_neurons + 1)]} for i in range(self.n_outputs)]
        neural_net.append(hidden_layer)
        neural_net.append(output_layer)
        for layer in neural_net:
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




            
        
   