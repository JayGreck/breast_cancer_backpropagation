import dense_layer as Dense_Layer

from error_backpropagation import Error_Backpropagation

from activation import Activation
import numpy as np
class Train_Network:

    def sigmoid_transfer(self, neuron_output):
        # Calculates the derivative (the slope on a curve) of a neurons output
        return neuron_output * (1.0 - neuron_output)


    def forward_pass(self, neural_net, row):
        inputs = row
        for layer in neural_net:
            new_inputs = [] # Inputs from each layer
            for neuron in layer: # Taking activation output from each neuron in the layer
                
                activation = self.activation.activate_neuron(inputs, neuron['weights'])
                neuron['output'] = self.sigmoid_transfer(activation) # Activation function
                new_inputs.append(neuron['output']) # inputs for next layer
            inputs = new_inputs
        return inputs

    def __init__(self, neural_net):
        self.activation = Activation() # Initialising weights and inputs
        self.backpropagate_err = Error_Backpropagation(neural_net)
        self.neural_net = neural_net

        print("Inside Train Network Constructor")
        
        
    
    def test(self, train, learning_rate, amt_epochs, n_outputs, neural_net):
        

        for epoch in range(amt_epochs):
            sum_error = 0

            for row in train:
                
                outputs = self.forward_pass(neural_net, row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1
                print("Outputs " + str(outputs))
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                print("Sum Error Print: " + str(sum_error))
                
                self.backpropagate_err.backward_propagate_error(expected, neural_net)
                self.backpropagate_err.update_weights(row, learning_rate, neural_net)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))