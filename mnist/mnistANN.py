import cPickle, gzip, numpy as np
import neuralNetwork as nn

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train, valid, test = cPickle.load(f)
f.close

features = train[0].shape[1]
obs = train[0].shape[0]
labels = np.ptp(train[1])+1 

trainFeatures = np.transpose(train[0])
trainLabels = np.zeros( (labels, obs) )

for i in range(obs):
	trainLabels[train[1][i]][i] = 1

ann = nn.neuralNetwork(trainFeatures, trainLabels, 10)
ann.normalizeInputs()
ann.initializeWeights()
#ann.train(100)

