import numpy as np
class Activation:

    def __init__(self):
        # self.weights = weights
        # self.inputs = inputs
        # self.activation = weights[-1] # Bias?
        # self.sigmoid_activation_func = lambda x : np.exp(x) / (1 + np.exp(x))
        pass
    print("Inisde Activation Constructor")
            
    def activate_neuron(self, inputs, weights):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += inputs[i] * weights[i] # np.dot(weights[i], inputs[i])
        return activation

    def sigmoid_activation(self, activation):
        return 1.0 / (1.0 + np.exp(-activation))
    
        