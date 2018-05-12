import numpy as np

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def derivative_tanh(x):
    return (1-tanh(x))*(1+tanh(x))

class NeuralNetwork:
    def __init__(self,arch):
        '''Initialize a neural network, giving the architecture of NN by spicifying
# of neurons at each layerself.
Example : NN = NeuralNetwork(arch=[2,4,1]) is a neural network with 2 neuron at
Input layer, 1 neuron at output layer and a hidden layer with 4 neron.'''
        self.active_function = tanh #For simplicity,let's hardcode it as tanhself.
        self.d_active_function = derivative_tanh
        #Randomly assign values between -1 and 1 to weights at each neuronself.
        self.weights = []
        for i,j in zip(arch,arch[1::]):
            self.weights.append(np.random.rand(j,i) + np.random.randint(-1,1,size=(j,i)))
    def predict_batch(self,x_data,x_label):
        for data,label in zip(x_data,x_label):
            vals = data
            for ns in self.weights:
                vals = self.active_function(s.dot(vals))
            print("Prediction : ",vals,"Answer : ",label)

    def predict(self,x_data):
        vals = x_data
        for ns in self.weights:
            vals = self.active_function(ns.dot(vals))
        return vals
#    def train_batch(self,x_data,x_label,lrt=0.1,epoch=100,batch_size=32):
#        for label,data in zip(x_data,x_label):
