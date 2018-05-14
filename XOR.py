import matplotlib.pyplot as plt
from NN_def import *



XOR = NeuralNetwork(np.array([2,3,4,1]))
XOR.train_batch(np.array([[0,1],[1,1],[1,0],[0,0]]),np.array([[1],[0],[1],[0]]),lrt=0.1,epoch=10000,batch_size=250)
