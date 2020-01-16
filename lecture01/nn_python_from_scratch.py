import numpy as np 

class NeuralNetwork(object):

    def __init__(self, layers = [1, 10, 1], activations=['sigmoid', 'sigmoid']):

        self.layers = layers
        self.activations = activations

        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i+1], layers[i]))
            self.biases.append(np.random.randn(layers[i+1], 1))

        self._weights_grad = [0 for _ in range(len(layers)-1)]
        self._biases_grad = [0 for _ in range(len(layers)-1)]
        # self._z_s = []
        # self._a_s = []

    def init_wb_grad(self, ):
        '''
        '''
        self._weights_grad = [0 for _ in range(len(self.layers)-1)]
        self._biases_grad = [0 for _ in range(len(self.layers)-1)]


    def feedforward(self, x):
        '''
        '''
        a = np.copy(x)
        z_s = []
        a_s = [a]

        for i in range(len(self.weights)):
            activation_fn = self.get_ActivationFunc(self.activations[i])
            z = self.weights[i].dot(a) + self.biases[i]
            a = activation_fn(z)
            z_s.append(z)
            a_s.append(a)
        
        self._z_s = z_s
        self._a_s = a_s

        return z_s, a_s


    def backpropagation(self, y, z_s=None, a_s=None):
        '''
        '''
        deltas = [None for _ in range(len(self.weights))]  # error delta = dC/dZ

        # loss: y - a_s[-1]
        deltas[-1] = ((y - self._a_s[-1]) * self.getDerivitiveActivationFunc(self.activations[-1])(self._z_s[-1]))

        for i in reversed(range(len(deltas)-1)):
            deltas[i] = self.weights[i+1].T.dot(deltas[i+1]) * self.getDerivitiveActivationFunc(self.activations[i])(self._z_s[i])

        n = y.shape[1]
        db = [d.dot(np.ones((n, 1), dtype=np.float)) / n for d in deltas]
        dw = [d.dot(self._a_s[i].T) / n for i, d in enumerate(deltas)]

        self._weights_grad = dw
        self._biases_grad = db

        return dw, db


    def update(self, lr=1e-2):
        '''
        '''
        self.weights = [w + lr * dw for w, dw in zip(self.weights, self._weights_grad)]
        self.biases = [b + lr * db for b, db in zip(self.biases, self._biases_grad)]
        # self._weights_grad = [0 for _ in range(len(self.layers)-1)]
        # self._biases_grad = [0 for _ in range(len(self.layers)-1)]


    @staticmethod
    def get_ActivationFunc(name):
        '''
        '''
        if name == 'sigmoid':
            return lambda x: np.exp(x) / (1 + np.exp(x))
        elif name == 'relu':
            def relu(x):
                y = np.copy(x)
                y[y<0] = 0
                return y 
            return relu
        elif name == 'linear':
            return lambda x: x
        else:
            pass

    @staticmethod
    def getDerivitiveActivationFunc(name):
        '''
        '''
        if name == 'sigmoid':
            sig = lambda x: np.exp(x) / (1 + np.exp(x))
            return lambda x: sig(x) * (1 - sig(x))
        elif name == 'linear':
            return lambda x: 1
        elif name == 'relu':
            def relu_diff(x):
                y = np.copy(x)
                y[y>=0] = 1
                y[y<0] = 0
                return y
            return relu_diff

    def train(self, x, y, epoches=100, batch_size=0, lr=0.01):
        '''
        '''
        for _ in range(epoches):
            i = np.random.randint(low=0, high=batch_size)
            while i < y.shape[-1] - batch_size:
                x_batch = x[:, i: i + batch_size]
                y_batch = y[:, i: i + batch_size]
                i += batch_size
                _, a_s = self.feedforward(x_batch)
                dws, dbs = self.backpropagation(y_batch)
                # self.update(lr=lr)
                self.weights = [w + lr * dw for w, dw in zip(self.weights, dws)]
                self.biases = [b + lr * db for b, db in zip(self.biases, dbs)]

                print("loss={}".format(np.linalg.norm(a_s[-1] - y_batch)))



if __name__=='__main__':
    import matplotlib.pyplot as plt
    nn = NeuralNetwork([1, 100, 1], activations=['sigmoid', 'sigmoid'])
    X = 2 * np.pi * np.random.rand(1000).reshape(1, -1)
    y = np.sin(X)

    nn.train(X, y, 10000, 64, 0.1)

    _, a_s = nn.feedforward(X)
    #print(y, X)
    plt.scatter(X.flatten(), y.flatten())
    plt.scatter(X.flatten(), a_s[-1].flatten())
    plt.show()