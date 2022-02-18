import numpy as np




class Error_Backpropagation:

    def __init__(self, neural_net):
        print("Inisde Error Backpropagation Constructor")
        #self.neuron_output = neuron_output
        self.neural_net = neural_net
        
        self.error = 0.0
        self.neuron_errors = list() # a list for the errors

       
    
     # Method to update weights after the error backpropagation has been performed
    def update_weights(self, layer, learning_rate, neural_net):
        for i in range(len(neural_net)):
            neuronal_inputs = layer[:-1] # Excluding the output layer?
            if i != 0:
                for neuron in neural_net[i-1]:
                    neuronal_inputs = [neuron['output']]
            for neuron in neural_net[i]:
                for z in range(len(neuronal_inputs)):
                    neuron['weights'][z] -= learning_rate * neuron['error_signal'] * neuronal_inputs[z]
                neuron['weights'][-1] -= learning_rate * neuron['error_signal']
        

    def sigmoid_transfer(self, neuron_output):
        # Calculates the derivative (the slope on a curve) of a neurons output
        return neuron_output * (1.0 - neuron_output)

    # Calculate the error for each neuron output
    def backward_propagate_error(self, expected, neural_net):
        for i in reversed(range(len(neural_net))): # Reversing so that the error calculations are going from ouput -> hidden layer
            network_layer = neural_net[i]
            neuron_errors = list()
            if i != len(neural_net)-1: # If i not the output layer
                for z in range(len(network_layer)):
                    neuron_error = 0.0
                    for neuron in neural_net[i+1]:
                        neuron_error += (neuron['weights'][z] * neuron['error_signal'])
                        #print("Neuron error !!!!: " + str(neuron_error))
                        
                    neuron_errors.append(neuron_error)
            else: # Calculate each output neuron error
                for z in range(len(network_layer)):
                    neuron = network_layer[z]
                   
                    neuron_errors.append(neuron['output'] - expected[z])

            for z in range(len(network_layer)):
                neuron = network_layer[z]
                neuron['error_signal'] = neuron_errors[z] * self.sigmoid_transfer(neuron['output']) # The error signal 



