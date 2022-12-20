import numpy as np
from numpy.random import rand

np.random.seed(0)


class Network:
    def __init__(self, n_input, n_output, lr=0.01):
        self.n_input = n_input  # input size
        self.n_output = n_output  # output size

        self.lr = lr

        # Define weights and biases
        self.w1 = rand(self.n_input, self.n_output)
        self.b1 = rand(1)

    def forward(self, x):
        """To Do: implement me"""
        self.h = self.w1 @ x + self.b1
        self.sigmoid(self.h)
        self.sigmoid_gradient(self.s)
        pass

    def sigmoid(self, x):
        """To Do: implement me"""

        self.s = 1/(1+np.exp(-x))
        pass

    def sigmoid_gradient(self, x):
        """To Do: implement me"""
        self.sg = (1-x) * x
        pass

    def loss(self, y, y_hat):
        """To Do: implement me"""
        self.l =  (0.5*(y_hat-y)**2).sum()
        pass

    def backward(self, x, y, h):
        """To Do: implement me"""
        self.delta_w1 = (self.s - y)
        # delta_g = (self.)
        pass

if __name__ == "__main__":
    import numpy as np
    from numpy.random import rand

    np.random.seed(0)  # DO NOT CHANGE

    x = rand(2,1) # dummy input
    y = rand(2,1) # dummy y / ground truth

    print("x = " + str(x))
    print("y = " + str(y))
    #initialize the network
    network = Network(2, 2)
    
    #run the forward pass on x and print the result
    h, forward_out = network.forward(x)
    print(h)
    print(forward_out)

    # compute loss
    LOSS = network.loss(y_hat=forward_out, y=y)
    print('Loss: {}'.format(LOSS))

    # Perform backward pass
