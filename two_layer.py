# a basic 2 layer neural net that solves: x3 && xor(x1, x2)
# activation function: sigmoid
# loss function: sum of squares error Sigma (y-y`)**2
import numpy as np
import matplotlib
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt

iter = 2000


def sigmoid(x):
    return 1./(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1.-x)

class NeuralNetwork:
# setup
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4)
        self.weights2   = np.random.rand(4,1)
        self.y          = y
        self.output     = np.zeros(y.shape)

    # Feeding forward
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

x = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y = np.array([[0], [1], [1], [0]])
nn = NeuralNetwork(x,y)


loss = np.zeros(iter)

# training
for i in range(iter):
    nn.feedforward()
    nn.backprop()
    loss[i] = (0-nn.output[0][0])**2 + (1-nn.output[1][0])**2 + (1-nn.output[2][0])**2 + (0-nn.output[3][0])**2


# save state - change matrices to list (can't store matrix w/ pyyaml)
state = {
    'input': nn.input.tolist(),
    'weights1': nn.weights1.tolist(),
    'weights2': nn.weights2.tolist(),
    'y': nn.y.tolist(),
    'output': nn.output.tolist()
}
with open('state1.yaml', 'w') as output_f:
    yaml.dump(state, output_f, default_flow_style=True)

# load state - change list's back to matrices
with open('state1.yaml', 'r') as stream:
    try:
        loaded = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

loaded['input'] = np.array(loaded['input'])
loaded['weights1'] = np.array(loaded['weights1'])
loaded['weights2'] = np.array(loaded['weights2'])
loaded['y'] = np.array(loaded['y'])
loaded['output'] = np.array(loaded['output'])

# plotting results
plt.plot(loss)
plt.savefig('fig1')
#plt.xticks(np.arange(0, iter, step=100))
#plt.show()

# print final result
print(nn.output)
