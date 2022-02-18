
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process


class Train_Network:

    def __init__(self, inputs, targets, l_rate, epochs, neural_net):
        
        # Lists used for plotting the MSE vs Epoch
        self.MSE_list = MSE_list = []
        self.epoch_list = epoch_list = []
        for i in range(epochs):
            sum_error = 0

            for input, target in zip(inputs, targets):

                
                
                # Forward pass
                output = neural_net.forward_pass(input)
                
                # Calculate error
                error = target - output
                    

                neural_net.backpropagation_obj.backpropagate(error)

                # Update weights and biases (gradient descent)
                neural_net.backpropagation_obj.update_parameters(l_rate)
               
                # Each input, calculaute sum error
                sum_error += self.mean_squared_error(target, output)
                
            epoch_error = sum_error / len(inputs)
            
            self.MSE_list.append(epoch_error)
            self.epoch_list.append(i)
            # Displays error at end of an epoch
            print("At Epoch: {}  MSE = {}".format(i, epoch_error))
        
       
        
        
    def mean_squared_error(self, target, output):
        return np.average((target - output)**2)
    
    def network_training_curve(self):
        plt.plot(self.epoch_list, self.MSE_list)
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.title("MSE vs Epoch")
        plt.legend(['Training'])
        plt.show()