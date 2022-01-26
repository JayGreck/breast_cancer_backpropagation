import numpy as np
class Activation:

    def __init__(self, inputs, weights):
        self.weights = weights
        self.inputs = inputs
        self.activation = weights[-1] # Bias?
            
    def activate_neuron(self):
        for i in range(len(self.weights)-1):
            self.activation += np.dot(self.weight[1], self.inputs[i]) # self.inputs[i] * self.weights[i]
        return self.activation

    def sigmoid_activation(self):
        return 1.0 / (1.0 + np.exp(-self.activation))
    
        