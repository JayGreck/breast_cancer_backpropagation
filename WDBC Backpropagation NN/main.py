from Dense_Layer import Dense_Layer
from Train_Network import Train_Network
from Preprocess_Data import Preprocess_Data



# Create a Network
network = Dense_Layer(30, [15], 2)


X_train, X_test, y_train, y_test = Preprocess_Data().get_dataframe()

y_train = y_train.to_numpy() # Testing Data

X_train = X_train.to_numpy() # Training Data

X_test = X_test.to_numpy().flatten() # Targets

y_test = y_test.to_numpy().flatten() # Targets


# train network
train = Train_Network(X_train, X_test, 0.5, 3000, network)

# Test the network
test = network.test_network(y_train, y_test, network)

# Plot Training Curve
train.network_training_curve()

