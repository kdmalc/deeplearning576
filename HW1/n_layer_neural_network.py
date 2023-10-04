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
        # Passing in "self" (current neural network obj) so I can access activation funcs and such
        # Set the 1st layer as input_dim x hidden_dim
        self.layers.append(Layer(self, nn_input_dim, nn_hidden_dim, actFun_type))
        # Set the middle layers as hidden_dim x hidden_dim
        for _ in range(n_layers - 3):
            self.layers.append(Layer(self, nn_hidden_dim, nn_hidden_dim, actFun_type))
        # Set the last layer as hidden_dim x output_dim
        self.layers.append(Layer(self, nn_hidden_dim, nn_output_dim, 'softmax'))

    def feedforward(self, X):
        # YOU IMPLEMENT YOUR feedforward HERE
        self.activations = []
        # First "activation" is just the input to the network
        activation = X
        for layer in self.layers:
            # Activations of each previous layer are fed into the next
            layer.feedforward(activation) # Sets layer.a, AKA the resulting activation
            activation = layer.a
            self.activations.append(activation)
        # Final activation is the output of softmax and thus the predicted probabilities
        self.probs = activation
        return None

    def calculate_loss(self, X, y):
        num_examples = len(X)
        self.feedforward(X)
        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        y_onehot = OneHotEncoder(sparse_output=False).fit_transform(y.reshape((-1, 1)))
        data_loss = (-1/num_examples) * np.sum(np.log(self.probs) * y_onehot)
        # Add regulatization term to loss (optional)
        data_loss += (self.reg_lambda / 2) * np.sum([np.square(layer.W).sum() for layer in self.layers])
        return (1. / num_examples) * data_loss

    def predict(self, X):
        # This is DNN feedfoward (eg not layer.feedfoward):
        self.feedforward(X) # This sets self.probs
        # argmax is the class with the highest predicted probability
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        # IMPLEMENT YOUR BACKPROP HERE
        num_examples = len(X)
        # delta3 is the output of the neural network, adjusted for predicting what the correct class is
        delta3 = self.probs.copy()
        delta3[range(num_examples), y] -= 1
        # Use Layer()-wise backprop
        # Derivative of the activation
        da = self.layers[-1].backprop(da=None, a_previous=self.layers[-2].a, dz=delta3)
        # Iterate through [n_layers-3, n_layers-4, ... , 1]
        for i in np.arange(1, self.n_layers-2)[::-1]:
            # Note: layer.backprop DOES return something, but dnn.backprop does NOT
            da = self.layers[i].backprop(da=da, a_previous=self.layers[i-1].a)
        self.layers[0].backprop(da, X)
        return None

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        # Eg run gradient descent for num_passes iters
        for i in range(0, num_passes):
            self.feedforward(X)
            self.backprop(X, y)
            for layer in self.layers:
                layer.gradient_step(epsilon, self.reg_lambda)
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))
                print("Accuracy after iteration %i: %f" % (i, (self.probs.argmax(axis=1) == y).sum() / len(y)))

    def visualize_decision_boundary(self, X, y):
        tlnn.plot_decision_boundary(lambda x: self.predict(x), X, y)

class Layer(object):
    def __init__(self, neuralnet_obj, nn_input_dim, nn_output_dim, actFun_type='tanh'):
        self.neuralnet_obj = neuralnet_obj # Required in order to access self.activationfuncs defined in dnn class
        self.nn_input_dim = nn_input_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        # Init weight and biases to random matrix and vector of zeros, respectively
        self.W = (1/np.sqrt(self.nn_input_dim)) * np.random.randn(self.nn_input_dim, self.nn_output_dim) 
        self.b = np.zeros((1, self.nn_output_dim))
        
    # Implement feedforward for a single layer
    def feedforward(self, X):
        self.z = X @ self.W + self.b
        self.a = self.neuralnet_obj.actFun(self.z, self.actFun_type)
        return None
    
    # Implement backprop for a single layer
    def backprop(self, da, a_previous, dz=None):
        # z is the output
        # a is the actual activation
        # W, b are the weights and biases
        if dz is None:
            self.dz = self.neuralnet_obj.diff_actFun(self.z, self.actFun_type) * da
        else:
            self.dz = dz
        self.dW = a_previous.T @ self.dz
        self.db = np.sum(self.dz, axis=0, keepdims=True)
        return self.dz @ self.W.T
                                            
    def gradient_step(self, epsilon, reg_lambda):
        self.dW += reg_lambda * self.W
        self.W -= epsilon * self.dW
        self.b -= epsilon * self.db
        return None

if __name__ == "__main__":
    # generate and visualize Make-Moons dataset
    X, y = tlnn.generate_data()
    #plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    #plt.show()

    actFun_type = 'tanh' #sigmoid, relu
    hidden_neurons_per_layer = 10
    num_layers = 5

    deep_model_tanh_5_10 = DeepNeuralNetwork(nn_input_dim=2, nn_hidden_dim=hidden_neurons_per_layer, nn_output_dim=2, n_layers=num_layers, actFun_type=actFun_type)
    deep_model_tanh_5_10.fit_model(X, y, num_passes=10000)
    plt.title(f'{actFun_type}, {num_layers} layers of size {hidden_neurons_per_layer}') 
    deep_model_tanh_5_10.visualize_decision_boundary(X,y)