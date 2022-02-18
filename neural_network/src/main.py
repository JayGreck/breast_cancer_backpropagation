
from random import seed
from dense_layer import Dense_Layer
from preprocess_data import Preprocess_Data
from error_backpropagation import Error_Backpropagation
from train_network import Train_Network
 

# Getting data
#dataframe = Preprocess_Data()
#print(dataframe.get_dataframe())


# Initialising Neural Network
#network = Dense_Layer(30, 15, 2)

#network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		#[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
#row = [1, 0, None]
#output = init_network.forward_pass(network, row)
#print(output)

# network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
# 		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
# expected = [0, 1]
# test = Error_Backpropagation(network, expected)
# test.backward_propagate_error()
# for layer in network:
# 	print(layer)



seed(1)

dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = Dense_Layer(n_inputs, 2, n_outputs)
n2 = network.neural_net_func()
train = Train_Network(n2)
train.test(dataset, 0.5, 20, n_outputs, n2)
for layer in n2:
	print(layer)