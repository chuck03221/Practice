import matplotlib.pyplot as plt
from NN_def import *



XOR = NeuralNetwork(np.array([2,4,1]))
print(XOR.predict(np.array([0,1])))
