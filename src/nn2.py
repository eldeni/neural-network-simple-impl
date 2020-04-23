import numpy as np

debug = True
def log(msg = ''):
  if debug is True:
    print(msg)

# training dataset
# X = (hours sleeping, hours studying), y = score on test
rawX = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
rawY = np.array(([92], [86], [89]), dtype=float)

# scaled values
X = rawX / np.amax(rawX, axis=0) # maximum of X array
y = rawY / 100 # max test score is 100

log('rawX: %s, amax(0): %s, scaledX: %s' % (rawX.tolist(), np.amax(rawX, axis=0).tolist(), X.tolist()))
log('rawY: %s, scaledY: %s' % (rawY.tolist(), y.tolist()))

class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 2
    self.outputSize = 1
    self.hiddenSize = 3

    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
    log('Neural_Network.W1: %s' % (self.W1.tolist()))
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer
    print('Neural_Network.W2: %s' % (self.W2.tolist()))

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    log('Neural_Network.forward(): z: %s, z2: %s, z3: %s' % (self.z.tolist(), self.z2.tolist(), self.z3.tolist()))
    o = self.sigmoid(self.z3) # final activation function
    return o

  def sigmoid(self, s):
    # activation function
    return 1 / (1 + np.exp(-s))

  # slope function
  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propgate through the network
    log('Neural_Network.backward(): y: %s, o: %s' % (y.tolist(), o.tolist()))

    self.o_error = y - o # error in output
    self.o_delta = self.o_error * self.sigmoidPrime(o) # applying derivative of sigmoid to error
    log('Neural_Network.backward(): o_error: %s, o_delta: %s' % (self.o_error.tolist(), self.o_delta.tolist()))

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error
    log('Neural_Network.backward(): z2_error: %s, z2_delta: %s' % (self.z2_error.tolist(), self.z2_delta.tolist()))

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights
    log('Neural_Network.backward(): adjusted weights, w1: %s, w2: %s, ' % (self.W1.tolist(), self.W2.tolist()))

  def train (self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

def main():
  NN = Neural_Network()
  log()

  # training operation
  for i in range(1): # trains the NN 1,000 times
    log('training: (%s)th iteration' % (i))

    prediction = NN.forward(X)
    log('input: %s' % (X.tolist()))
    log('output (ground truth): %s' % (y.tolist()))
    log('prediction: %s' % (prediction.tolist()))
    log('loss: normalizedLoss: %s, rawLoss: %s, rawLossSquared: %s' % (
      np.mean(np.square(y - prediction).tolist()),
      (y - prediction).tolist(),
      np.square(y - prediction).tolist()
    )) # mean sum squared loss
    NN.train(X, y)
    log()

  test1 = np.array(([2, 9]), dtype=float)
  print('test: %s' % (test1.tolist()))
  print('test prediction: %s' % (NN.forward(test1).tolist()))

if __name__ == "__main__":
    main()
