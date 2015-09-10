import numpy as np

class neuralNetwork():
        def __init__(self, features, targets, hiddenNodes):
                self.i = features.shape[0]
                self.j = hiddenNodes
                self.k = targets.shape[0]

                self.obs = features.shape[1]
                self.inputs = features
                self.outputs = targets

        def normalizeInputs(self):
                self.inputs = (self.inputs - np.mean(self.inputs))/ np.std(self.inputs)
                self.inputs = np.r_[self.inputs, 0.9*np.ones((1, self.obs))]
        def initializeWeights(self):
                self.Wji = np.random.rand(self.j, self.i+1) - 0.5
                self.Wkj = np.random.rand(self.k, self.j+1) - 0.5

        def actFunk(self, x):
                return np.tanh(x)

        def delFunk(self, x):
                return 1 - x ** 2

        def train(self, iterations):
                for i in range(iterations):
                        h = np.r_[self.actFunk(np.dot(self.Wji, self.inputs)), 0.9*np.ones((1, self.obs))]
                        y = self.actFunk(np.dot(self.Wkj, h))

                        err = (self.outputs - y)

                        delK = 0.1 * err * self.delFunk(y);
                        delJ = np.transpose(np.dot(np.transpose(delK), self.Wkj)) * self.delFunk(h)
                        deltaK = np.dot(delK, np.transpose(h))
                        deltaJ = np.dot(delJ, np.transpose(self.inputs))

                        self.Wkj = self.Wkj + deltaK / self.obs
                        self.Wji = self.Wji + deltaJ[0:self.j,:] / self.obs

                        print np.mean(np.mean(err))

        def test(self, features, labels):
                features = (features - np.mean(features))/ np.std(features)
                features = np.r_[features, 0.9*np.ones((1, self.obs))]
                h = np.r_[self.actFunk(np.dot(self.Wji, features)), 0.9*np.ones((1, self.obs))]
                y = self.actFunk(np.dot(self.Wkj, h))

                return y

