import numpy as np
np.random.seed(0)
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
        self.arch = arch
        #Randomly assign values between -1 and 1 to weights at each neuronself.
        self.weights = []
        for i,j in zip(arch,arch[1::]):
            self.weights.append(np.random.rand(j,i) + np.random.randint(-1,1,size=(j,i)))

    def predict(self,x_data):
        vals = x_data
        for ns in self.weights:
            vals = self.active_function(ns.dot(vals)+1)
        return vals
    def train_batch(self,x_data,x_label,lrt=0.1,epoch=1000,batch_size=32):
        for epoch_c in range(epoch):
            for data,label in zip(x_data,x_label):
                vals = data
                tmp = [data]
                back_prop = []
                for ns in self.weights:
                    vals = ns.dot(vals)+1
                    tmp.append(vals)
                    vals = self.active_function(vals)
                back_prop_base = lrt*(label-vals)*self.d_active_function(tmp[-1])
                for w,y in zip(self.weights[::-1],tmp[-2::-1]):
                    back_prop.append([base*y for base in back_prop_base])
                    back_prop_base = (back_prop_base.reshape(-1,1)*w).sum(axis=0)*self.d_active_function(y)
                self.weights = [old+delta for old,delta in zip(self.weights,back_prop[::-1])]
            if epoch_c%batch_size==0:
                print(np.hstack([self.predict(test) for test in x_data]))
                print("++++++++One Batch+++++++++++++")

                    




