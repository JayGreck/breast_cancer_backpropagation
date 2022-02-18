
import numpy as np

import matplotlib.pyplot as plt

class Train_Network:

    def __init__(self, inputs, targets, l_rate, epochs, neural_net):
        
        MSE = []
        epoch_list = []
        for i in range(epochs):
            sum_error = 0

            for input, target in zip(inputs, targets):

                
                
                # Forward pass
                output = neural_net.forward_pass(input)
                #print(output)
                
                
                #print("Output = " + str(output))
                    
                # Calculate error
                #print("target" + str(target))
                error = target - output
                    

                neural_net.backpropagation_obj.backpropagate(error)

                # Update weights and biases (gradient descent)
                neural_net.backpropagation_obj.update_parameters(l_rate)
               
                # Each input, calculaute sum error
                sum_error += self.mean_squared_error(target, output)
                
            epoch_error = sum_error / len(inputs)
            
            MSE.append(epoch_error)
            epoch_list.append(i)
            # display error at end of an epoch
            print("At Epoch: {}  MSE = {}".format(i, epoch_error))
        
        plt.plot(epoch_list, MSE)
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.title("MSE vs Epoch")
        plt.legend(['Training'])
        plt.show()


    def mean_squared_error(self, target, output):
        return np.average((target - output)**2)