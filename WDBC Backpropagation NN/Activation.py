import numpy as np

class Activation:

    def __init__(self):
        pass

    def sigmoid_activation(activation):
        return 1.0 / (1.0 + np.exp(-activation))
    
    def sigmoid_transfer(output):
        # Calculates the derivative (the slope on a curve) of a neurons output
        return output * (1.0 - output)
    
   