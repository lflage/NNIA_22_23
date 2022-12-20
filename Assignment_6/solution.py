import numpy as np
from numpy.random import rand


class Network:
    def __init__(self, n_input, n_output, lr=0.01):
        self.n_input = n_input  # input size
        self.n_output = n_output  # output size

        self.lr = lr

        # Define weights and biases
        self.w1 = rand(self.n_input, self.n_output)
        self.b1 = rand(1)

    def forward(self, x):
        self.h = self.w1@x+self.b1
        self.sigmoid(self.h)
        self.sigmoid_gradient(self.s)
        print('Before activation\n',self.h,'\n')
        print('Sigmoid activation\n',self.s,'\n')
        print('Sigmoid gradient\n',self.sg,'\n')
        
    def sigmoid(self, x):
        self.s = 1/(1+np.exp(-x))

    def sigmoid_gradient(self, x):
        self.sg = x*(1-x)
        

    def loss(self, y, y_hat):
        self.l = 1/2*((y_hat-y)**2)
        print('Loss\n',self.l,'\n')

    def backward(self, x, y, h):
        # NOTE: this implementation is not using h or the loss method computed before
        # since we were not sure how the computations should then look like
        # Instead this is a reconstruction of tha manual backpropagation from 2.3-2.5
        # using elementwise multiplication for the loss gradient and the sigmoid gradient
        # and matrix multiplication with the transposed input x
        
        y_hat = self.s
        self.dw1 = (y_hat-y)*self.sg@x.T
        self.db1 = ((y_hat-y)*self.sg).T@np.array([[1],[1]])
        print('Delta weights\n',self.dw1,'\n')
        print('Delta bias\n',self.db1,'\n')
        
    

#np.random.seed(0)

#x = rand(2,1) # dummy input
#y = rand(2,1) # dummy y / ground truth

#print("x = " + str(x))
#print("y = " + str(y))
#print('\n')

#initialize the network
#network = Network(2, 2)
#network.forward(x)
#network.loss(y,network.s)
#network.backward(x,y,network.h)

