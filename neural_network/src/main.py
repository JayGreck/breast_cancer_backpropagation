from nbformat import read
from init_network import init_network
from read_data import read_data

# Getting data
dataframe = read_data()
dataframe.get_dataframe()

# Initialising Neural Network
#network = init_network(30, 15, 2)