'''
Write your own n layer neural network.py that builds and trains a neural network of n layers. 
Your code must be able to accept as parameters (1) the number of layers and (2) layer size.

1. Create a new class, e.g DeepNeuralNetwork, that inherits NeuralNetwork in three_layer_neural_network.py
2. In DeepNeuralNetwork, change function feedforward, backprop, calculate loss and fit model
3. Create a new class, e.g. Layer(), that implements the feedforward and backprop steps for a single layer in the network
4. Use Layer.feedforward to implement DeepNeuralNetwork.feedforward
5. Use Layer.backprop to implement DeepNeuralNetwork.backprop
6. Notice that we have L2 weight regularizations in the final loss function in addition to the cross entropy.
    Make sure you add those regularization terms in DeepNeuralNetwork.calculate loss and their derivatives in DeepNeuralNetwork.fit_model.
'''

import three_layer_neural_network as tlnn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

class DeepNeuralNetwork(tlnn.NeuralNetwork):
    def __init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim, n_layers, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units (layer size)
        :param nn_output_dim: output dimension
        :param n_layers: number of layers
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.n_layers = n_layers
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        
        # initialize the weights and biases in the network
        np.random.seed(seed)
        
        self.layers = []
        self.layers.append(self.Layer(self, nn_input_dim, nn_hidden_dim, actFun_type))
        for _ in range(n_layers - 3):
            self.layers.append(self.Layer(self, nn_hidden_dim, nn_hidden_dim, actFun_type))
        self.layers.append(self.Layer(self, nn_hidden_dim, nn_output_dim, 'softmax'))

    def feedforward(self, X):
        # YOU IMPLEMENT YOUR feedforward HERE
        self.activations = []
        activation = X
        for layer in self.layers:
            activation = layer.feedforward(activation)
            self.activations.append(activation)
        self.probs = activation
        return None

    def calculate_loss(self, X, y):
        num_examples = len(X)
        self.feedforward(X)
        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        y_onehot = OneHotEncoder(sparse_output=False).fit_transform(y.reshape((-1, 1)))
        data_loss = (-1/num_examples) * np.sum(np.log(self.probs) * y_onehot)
        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda / 2 * np.sum([np.square(layer.W).sum() for layer in self.layers])
        return (1. / num_examples) * data_loss

    def predict(self, X):
        # This is DNN feedfoward:
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        # IMPLEMENT YOUR BACKPROP HERE
        num_examples = len(X)
        delta3 = self.probs.copy()
        delta3[range(num_examples), y] -= 1
        # Use Layer()-wise backprop
        da = self.layers[-1].backprop(da=None, a_previous=self.layers[-2].a, dz=delta3)
        
        # Iterate through [n_layers-3, n_layers-4, ... , 1]
        for i in np.arange(1, self.n_layers-2)[::-1]:
            da = self.layers[i].backprop(da=da, a_previous=self.layers[i-1].a)
            
        self.layers[0].backprop(da, X)
        
        return None

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X)
            # Backpropagation
            self.backprop(X, y)
            
            for layer in self.layers:
                layer.step(epsilon, self.reg_lambda)

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))
                print("Accuracy after iteration %i: %f" % (i, (self.probs.argmax(axis=1) == y).sum() / len(y)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        tlnn.plot_decision_boundary(lambda x: self.predict(x), X, y)

class Layer():
    def __init__(self, dnn_instance, nn_input_dim, nn_output_dim, actFun_type='tanh'):
        self.dnn_instance = dnn_instance
        self.nn_input_dim = nn_input_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        # Init weight and biases to random matrix and vector of zeros, respectively
        self.W = np.random.randn(self.nn_input_dim, self.nn_output_dim) / np.sqrt(self.nn_input_dim)
        self.b = np.zeros((1, self.nn_output_dim))
        
    # Implement feedforward for a single layer
    def feedforward(self, X):
        self.z = X@self.W + self.b
        self.a = self.dnn_instance.actFun(self.z, self.actFun_type)
        return None
    
    # Implement backprop for a single layer
    def backprop(self, da, a_previous, dz=None):
        # z is the output (after the activation layer)
        # a is the actual activation
        # W, b are the weights and biases
        if dz is None:
            self.dz = self.dnn_instance.diff_actFun(self.z, self.actFun_type) * da
        else:
            self.dz = dz
        self.dW = a_previous.T.dot(self.dz)
        self.db = self.dz.sum(axis=0, keepdims=True)
        da_previous = self.dz.dot(self.W.T)
        return da_previous
                                            
    def update_weights(self, epsilon, reg_lambda):
        self.dW += reg_lambda * self.W
        self.W -= epsilon * self.dW
        self.b -= epsilon * self.db
        return None

if __name__ == "__main__":
    # generate and visualize Make-Moons dataset
    X, y = tlnn.generate_data()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

    actFun_type = 'tanh' #sigmoid, relu
    hidden_neurons_per_layer = 10
    num_layers = 5

    deep_model_tanh_5_10 = DeepNeuralNetwork(nn_input_dim=2, nn_hidden_dim=hidden_neurons_per_layer, nn_output_dim=2, n_layers=num_layers, actFun_type=actFun_type)
    deep_model_tanh_5_10.fit_model(X, y, num_passes=10000)
    plt.title(f'{actFun_type}, {n_layers} layers of size {nn_hidden_dim}') 
    deep_model_tanh_5_10.visualize_decision_boundary(X,y)