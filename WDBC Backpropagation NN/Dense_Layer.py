import numpy as np
from Activation import Activation
from Backpropagation import Backpropagation
import matplotlib.pyplot as plt

class Dense_Layer:

    def __init__(self, n_inputs, hidden_layers, n_ouputs):
        
        
        self.n_inputs = n_inputs
        self.hidden_layers = hidden_layers
        self.n_outputs = n_ouputs


        # Representation of the layers
        layers = [n_inputs] + hidden_layers + [n_ouputs]


        # Generating np array of zeroes for bias
        biases = list()
        for i in range(len(layers)):
            bias = np.zeros(layers[i]) # 1-Dimensional array of the amt of zeroes equal to the amt neurons in network
            biases.append(bias)
        self.biases = biases
        
        

        # Generate weighted connections
        weights = list()
        np.random.seed(2)
        for i in range(len(layers) - 1):
            weight = np.random.rand(layers[i], layers[i + 1])
            weights.append(weight)
        self.weights = weights
        

        # Generating np array of zeroes for activations
        activations = list()
        for i in range(len(layers)):
            init_activation_array = np.zeros(layers[i]) # 1-Dimensional array of the amt of zeroes equal to the amt neurons in network
            activations.append(init_activation_array)
        self.activations = activations # Equal to the amt of neurons in each layer
        

        # Generatign np array of zeroes for derivatives
        derivatives = list()
        for i in range(len(layers) - 1):
            derivative = np.zeros((layers[i], layers[i + 1])) # Outputs a matrix 
            derivatives.append(derivative)
        self.derivatives = derivatives
        
        self.backpropagation_obj = Backpropagation(self.derivatives, self.activations, self.weights, self.biases)
        

    # Calculates the forward pass of the neural net based on the input of previous layers
    def forward_pass(self, inputs):
       
        activations = inputs # Setting the activations as the inputs
        
        try:
            activations = float(activations)
            self.activations[0] = float(activations)
        except:
            pass
        
        # Iterate through the network layers
        for i, w in enumerate(self.weights):
                
                        
          
            # The dot product (input * weight) + bias
            dot_inputs = np.dot(activations, w) + self.biases[i][i]

            # Activation function
            activations = Activation.sigmoid_activation(dot_inputs) 
            
            # Storing activations at i+1 because activations are in the next layer for the multiplication to take place 
            self.activations[i + 1] = activations 
               
        return activations


    def test_network(self, inputs, targets, neural_net):
        pred_list = [] # List fo predictions
        true_positives = 0
        true_negatives = 0
        false_negatives = 0 # ppl who did have cancer but said they didn't
        false_positives = 0 # ppl who did not have cancer but said they did
        for i in range(1):
            

            for input, target in zip(inputs, targets):

                
                
                # Forward pass
                activations = neural_net.forward_pass(input)

                
                if activations[i] >= 0.5:
                    pred = 1
                else:
                    pred = 0
                
                # For Confusion Matrix
                if pred == 1 and target == 1:
                    true_positives += 1
                elif pred == 0 and target == 0:
                    true_negatives += 1
                elif pred == 1 and target == 0:
                    false_positives += 1
                elif pred == 0 and target == 1:
                    false_negatives += 1
                
                
                
                
                pred_list.append(pred)
        
        TN = true_negatives
        FP = false_positives
        FN = false_negatives
        TP = true_positives
        correct_instances = TP + TN

        print("\n")
        print("-------  CONFUSION MATRIX -------")
        print("0 = Benign")
        print("1 = Malignant")
        print("\n")
        print(str(TP) + " " + str(FP))
        print(str(FN) + " " + str(TN))
        print("\n")

        print("Test Accuracy: ", ((correct_instances/len(targets))*100), '%')
        print("Correctly Classified = " + str(correct_instances))
        print("Incorrectly Classified = " + str(FP + FN))
        print("False Positive Rate: ", FP/(FP+TN))
        print("True Positive Rate: ", TP/(TP+FN))
                
   