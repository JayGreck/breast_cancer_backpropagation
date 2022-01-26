

class Error_Backpropagation:

    def __init__(self, neuron_output):
        self.neuron_output = neuron_output

    def sigmoid_transfer(self):
        # Calculates the derivative (the slope on a curve) of a neurons output
        return self.neuron_output * (1.0 - self.neuron_output)
