from random import random

class init_network:

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