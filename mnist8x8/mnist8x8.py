import numpy as np
import neuralNetwork as nn
import scipy.io as sio

# Load the dataset
mat_contents = sio.loadmat('MNIST.mat')

# training data
train = np.transpose(mat_contents['Training'])
train = train.astype('double')
labels = np.transpose(mat_contents['TrainLabels'])
labels = labels.astype('double');
# testing data
test = np.transpose(mat_contents['Testing'])
test = test.astype('double')
tlabels = np.transpose(mat_contents['TestLabels'])
tlabels = tlabels.astype('double');
# validation data
val = np.transpose(mat_contents['Validation'])
val = train.astype('double')
vlabels = np.transpose(mat_contents['ValidationLabels'])
vlabels = vlabels.astype('double');

# const prms
features = train.shape[0]
obs = train.shape[1]
numlabels = np.ptp(labels)+1

# set labels to -1, 1; 0.95 has a better gradient
trainLabels = -0.95*np.ones( (numlabels, obs) )
for i in range(obs):
	trainLabels[labels[0][i]][i] = 0.95
trainFeatures = train

# train it
ann = nn.neuralNetwork(trainFeatures, trainLabels, 25)
ann.normalizeInputs()
ann.initializeWeights()
ann.train(1000)

# test it on the training set
# accuracy on the data it was trained off of
y = ann.test(trainFeatures)
trainGuess = np.argmax(y,axis=0)
AccTrain = np.sum( trainGuess==labels ).astype('double') / obs * 100
print AccTrain

# independent test data
y = ann.test(test)
testGuess = np.argmax(y,axis=0)
AccTest = np.sum( testGuess==tlabels).astype('double') / test.shape[1] * 100
print AccTest
