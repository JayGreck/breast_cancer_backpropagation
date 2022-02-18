

from Activation import Activation
import numpy as np



class Backpropagation:

    def __init__(self, derivatives, activations, weights, biases):
        self.derivatives = derivatives
        self.activations = activations
        self.weights = weights # Transposing the weights
        self.biases = biases
    

    def backpropagate(self, error):
        
        
        # Going from left to right by reversing the order
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]
            delta = error * Activation.sigmoid_transfer(activations)
            delta_T = np.transpose(delta.reshape(delta.shape[0], -1)) # Transposing the error signal (delta)
            current_activations = self.activations[i]
            current_activations_2D = current_activations.reshape(current_activations.shape[0], -1) # Converting to a vertical matrix (2D Array)

            self.derivatives[i] = np.dot(current_activations_2D, delta_T) # Calculates the derivative
            error = np.dot(delta, np.transpose(self.weights[i])) # transposing the weights at index i, and calculate the dot product of error signal delta dot weights
           
        return error
    
    # Updating the weights using gradient descent
    def update_parameters(self, l_rate):
        # Updating weights
        for i, z in zip(range(len(self.weights)), range(3)):
           
            # w is the weights
            w = self.weights[i]
           
            # d is the derivatives 
            d = self.derivatives[i]
            
            # Updates the weights
            w += l_rate * d
        
        # Updating biases
        z = 0
        while z < 3:
            
            b = len(self.biases)
            for p, u, j in zip(range(b), range(2), range(len(self.biases[z]))):
                b = self.biases[z][j] + l_rate * d
                self.biases[z][j] -= b[p][u]      
            z += 1
            

        
            

            
    

              
           
           
               
                
            
            

        