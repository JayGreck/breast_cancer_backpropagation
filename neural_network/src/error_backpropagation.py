import numpy as np




class Error_Backpropagation:

    def __init__(self, neural_net, expected):
        #self.neuron_output = neuron_output
        self.neural_net = neural_net
        self.expected = expected
        self.error = 0.0
        self.neuron_errors = list() # a list for the errors
        

    def sigmoid_transfer(self, neuron_output):
        # Calculates the derivative (the slope on a curve) of a neurons output
        return neuron_output * (1.0 - neuron_output)

    # Calculate the error for each neuron output
    def backward_propagate_error(self):
        for i in reversed(range(len(self.neural_net))): # Reversing so that the error calculations are going from ouput -> hidden layer
            network_layer = self.neural_net[i]
            if i != len(self.neural_net)-1: # If i not the output layer
                for z in range(len(network_layer)):
                    for neuron in self.neural_net[i+1]:
                        self.error += (np.multiply(neuron['weights'][z], neuron['error_signal'])) * self.sigmoid_transfer(neuron['output'])
                    self.neuron_errors.append(self.error)
            else: # Calculate each output neuron error
                for z in range(len(network_layer)):
                    neuron = network_layer[z]
                    self.neuron_errors.append((np.subtract(neuron['output'], self.expected)) * self.sigmoid_transfer(neuron['output'])) 

            for z in range(len(network_layer)):
                neuron = network_layer[z]
                neuron['error_signal'] = self.neuron_errors[z] # The error signal 



