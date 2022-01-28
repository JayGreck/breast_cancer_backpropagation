from nbformat import read
from dense_layer import Dense_Layer
from preprocess_data import Preprocess_Data
from error_backpropagation import Error_Backpropagation
 

# Getting data
dataframe = Preprocess_Data()
print(dataframe.get_dataframe())


# Initialising Neural Network
#network = Dense_Layer(30, 15, 2)

#network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		#[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
#row = [1, 0, None]
#output = init_network.forward_pass(network, row)
#print(output)

#network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		#[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
#expected = [0, 1]
#test = Error_Backpropagation(network, expected)
#test.backward_propagate_error()
#for layer in network:
	#print(layer)