

from Activation import Activation
import numpy as np



class Backpropagation:

    def __init__(self, derivatives, activations, weights, biases):
        self.derivatives = derivatives
        self.activations = activations
        self.weights = weights # Transposing the weights
        self.biases = biases
    

    def backpropagate(self, error, verbose=False):
        
        
        # Going from left to right by reversing the order
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]
            delta = error * Activation.sigmoid_transfer(activations)
            delta_T = np.transpose(delta.reshape(delta.shape[0], -1)) # Transposing the error signal (delta)
            current_activations = self.activations[i]
            current_activations_2D = current_activations.reshape(current_activations.shape[0], -1) # Converting to a vertical matrix (2D Array)

            self.derivatives[i] = np.dot(current_activations_2D, delta_T)
            error = np.dot(delta, np.transpose(self.weights[i])) #+ self.biases[i][i]

            # if verbose:
            #     print("Derivatives for W{}: {}".format(i, self.derivatives[i]))
        return error
    
    # Updating the weights using gradient descent
    def update_parameters(self, l_rate):
        for i, z in zip(range(len(self.weights)), range(3)):
           
            # w is the weights
            w = self.weights[i]
            #print("Original Weights{} {}".format(i, w))

           
            #print(self.biases[i][i]) 
            # np.transpose(b.reshape(b.shape[0], -1))
           
            # d is the derivatives 
            d = self.derivatives[i]
            
            
            # Updates the weights
            w += l_rate * d

            #print(self.weights)
            #print(len(self.biases[i]))
            #b += self.biases[i][i] + l_rate * d
        z = 0
        
        while z < 3:
            
            for j in range(len(self.biases[z])):
                #print(self.biases[z])
               # print(biases.shape)
                
                  

                
               
                b = self.biases[z][j] + l_rate * d
                
                
                #print("At " + str(j) + " biases = " + str(b))
                
                
                #b +=  l_rate * d
                # for p in range(len(b)):
                #     #print(p)

                #     for u in range(2):
                #         self.biases[z][j] = b[p][u]
                #print(self.biases[z][j])


                for p, u in zip(range(len(b)), range(2)):
                    #print(p)
                    self.biases[z][j] -= b[p][u]
                    # Used to be: biases[z][j] = b[p][u]
                    #print(biases[z][j])
                    #print(b.shape)
                    
            z += 1
            #b += l_rate * d

            
            
            #print(b)
            #b = b.reshape(b.shape[0], -1)
            
            # Update the biases
        #for r in range(self.biases):

        
            

            
    

              
           
           
               
                
            
            

            #print("Updated Weights{} {}".format(i, b))
        