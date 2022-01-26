

from audioop import reverse

from sklearn import neural_network


class Error_Backpropagation:

    def __init__(self, neuron_output, neural_net, expected):
        self.neuron_output = neuron_output
        self.neural_net = neural_net
        self.expected = expected
        self.error = 0.0
        

    def sigmoid_transfer(self):
        # Calculates the derivative (the slope on a curve) of a neurons output
        return self.neuron_output * (1.0 - self.neuron_output)

    # Calculate the error for each neuron output
    def backward_propagate_error(self):
        for i in reversed(range(len(self.nerual_net))): # Reversing so that the error calculations are going from ouput -> hidden layer
            network_layer = self.neural_net[i]
            neuron_errors = list() # a list for the errors
            if i != len(neural_network)-1: # If i not the output layer
                for z in range(len(network_layer)):
                    for neuron in self.neural_net[i+1]:
                        self.error += (neuron['weights'][z] * neuron['delta'])
                    neuron_errors.append(self.error)

