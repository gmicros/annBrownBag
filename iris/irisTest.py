#! /usr/bin/python
import numpy as np
import neuralNetwork as nn
import matplotlib.pyplot as plt

plt.close()

filename = 'iris.txt'
features = 4;
observations = len(open(filename).readlines())
classes = 3;

x = np.transpose(np.loadtxt(filename, delimiter=',', usecols=(0,1,2,3)))
y = np.transpose(np.loadtxt(filename, delimiter=',', usecols=(4,5,6)))

a = nn.neuralNetwork(x, y, 10);
a.normalizeInputs()
a.initializeWeights()
a.train(10000);

b = a.test(x, y)
plt.plot(b[0])
plt.plot(b[1])
plt.plot(b[2])