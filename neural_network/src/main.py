from nbformat import read
from init_network import init_network
from read_data import read_data
 

# Getting data
dataframe = read_data()
dataframe.get_dataframe()

# Initialising Neural Network
#network = init_network(30, 15, 2)

#network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		#[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
#row = [1, 0, None]
#output = init_network.forward_pass(network, row)
#print(output)