#-*- coding: utf-8 -*-
import numpy as np
import random

from collections import OrderedDict
from utils import numerical_gradient

import torch.nn as nn



def softmax(x):
        
    softmax_output = None
    
    # ========================================== WRITE YOUR CODE ========================================== #
    # Instructions: 
    #     Implement the softmax function
    #     Consider that the sum of the softmax output should be one
    #     Prevent the overflow (Do not let your function output the 'NaN')
    
    x = x - x.max(axis=-1, keepdims=True)
    y = np.exp(x)
    softmax_output = y / y.sum(axis=-1, keepdims=True)
    
    # ======================================================================================================
    return softmax_output




def cross_entropy_loss(score, target, thetas, lamb):
    
    
    delta = 1e-9
    batch_size = target.shape[0]
    CE_loss = 0
    reg_loss = 0
    loss = None
    
    # ========================================== WRITE YOUR CODE ========================================== #
    # Instructions: 
    #    Implement the cross entropy loss and regularization loss 
    #
    #    Cross Entropy term :
    #        Use delta to prevent the occurence of log(0)
    #        Consider the batch size(N)
    #
    #    Regularization term :
    #        Implement the L2 Regularization
    #        Use lamb as a regularization constant
    #        Multiply 0.5 to the reg_loss (for the computational convenience)
    
    sum_ = 0
    for value in thetas.values():
        sum_ += np.sum(np.square(value))
        
    CE_loss = -np.sum(target * np.log(score + 1e-9)) / batch_size

    reg_loss = (sum_)*(lamb/(2*batch_size))
    
    
    
    # ======================================================================================================
    
    loss = CE_loss + reg_loss
    
    return loss
    

    

class OutputLayer:
    
    
    def __init__(self, thetas, regularization):
        self.loss = None           # loss value
        self.output_softmax = None # Output of softmax
        self.target_label = None   # Target label (one-hot vector)
        self.thetas = thetas
        self.regularization = regularization
        
    def forward(self, x, y):
    
        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #    Compute the cross entropy loss using the softmax output
        #    Get the approriate input (score and target) for the loss function
        #    (Remember that the values in the forward propagation phase would be needed at the backward propagation phase)
               
        self.output_softmax = softmax(x)
        self.target_label = y
        self.loss = cross_entropy_loss(self.output_softmax, y, self.thetas, self.regularization)    
            
        
        # ======================================================================================================
    
        return self.loss
    
    def backward(self, dout=1):
        
        size = self.target_label.shape[0]
        dz = None
        
        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #    Since it is the output layer, the delta(dout) is one.
        #    Compute the backward propagation of the output layer
        #    (hint : Calculate the derivative of the loss with respect to the softmax output) 
        
        py = self.output_softmax[self.target_label == 1]
        D = np.zeros_like(self.output_softmax)
        D[self.target_label == 1] = -1/py.flat[0]
        d = D.flatten()
        s = -np.outer (self.output_softmax, self.output_softmax) + np.diag(self.output_softmax.flatten())
        dz = d.dot(s)

        # dz = self.output_softmax
        # dz[range(size), self.target_label] -= 1
        # dz = dz/size
        
        # ======================================================================================================
        
        return dz
    
    
class ReLU:
    
    
    def __init__(self):
        
        self.mask = None
        
    def forward(self, x):
        
        self.out = None
    
        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #    Implement ReLU function.
        #    All the negative values should be ignored
        #    Think which value to save for the backward propagation phase

        self.mask = (x>0).astype(np.int)
        self.out = np.maximum(x, 0)
        out = self.out
        
        
        # ======================================================================================================
    
        return out
    
    def backward(self, dout):
    
        dx = None
        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #    dout is the propagation value from the upper layer
        #    Only the points that had survived during the forward propagation should be backward propagated

        dx = dout * self.mask
        
        # ======================================================================================================
        
        return dx
    
class Sigmoid:
    
    
    def __init__(self):
        self.out = None
        
    def forward(self, x):
    
        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #    Implement sigmoid function
        #    Make sure that the output is in the range of 0 to 1

        self.out = 1 / (1 + np.exp(-x))
        out = self.out
        
        # ======================================================================================================
    
        return out
    
    def backward(self, dout):
        
        dx = None
        
        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #    dout is the propagation value from the upper layer
        #    Consider the derivative of the sigmoid function

        dx = dout * self.out * (1.0 - self.out)
        
        # ======================================================================================================
        
        return dx
    
    
    
class Affine:
    
    
    def __init__(self, T, b):
        self.Theta = T
        self.b = b
        self.x = None
        self.dT = None
        self.db = None
        
    def forward(self, x):
        
        out = None
    
        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #    Implement an Affine layer
        #    The input should be multiplied by weight and biases need to be considered
        #    Consider the backward propagation phase

        
        self.x = x
        out = x.reshape(x.shape[0], self.Theta.shape[0]).dot(self.Theta) + self.b
        
        # ======================================================================================================
    
        return out
    
    def backward(self, dout):
        
        dx = None
        
        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #    Compute the back-propagation value with respect to input x, weight W and bias b
        #    You should compute not only the output of the function dx, but also the derivative dW and db
        #    Return only the value of dx
        #    dT and db would be used at the weight updates
        
        dx = dout.reshape(-1, self.Theta.shape[1]).dot(self.Theta.T).reshape(self.x.shape)
        self.dT = self.x.T.dot(dout.reshape(self.x.shape[0], -1))
        self.db = np.sum(dout, axis=0)
        
        # ======================================================================================================
        
        return dx
    
    
    
    
from collections import OrderedDict
class TwoLayerNet:
    

    def __init__(self, input_size, hidden_size, output_size, theta_init_std = 0.01, regularization = 0.0):

        # Weight Initialization
        self.params = {}
        self.params['T1'] = theta_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['T2'] = theta_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        self.thetas = {}
        self.thetas['T1'] = self.params['T1']
        self.thetas['T2'] = self.params['T2']
        
        self.reg = regularization

        self.layers = OrderedDict()
        
        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #     Implement two layers net
        #     Model Structure would be like
        #       " Input => Fully Connected => ReLU => Fully Connected => OutputLayer "
        #     Use the classes(Affine, ReLU, OutputLayer) you had defined above
        

        self.layers['Affine1'] = Affine(self.params['T1'], self.params['b1'])
        self.layers['relu'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['T2'], self.params['b2'])
        
        # ======================================================================================================
        self.lastLayer = OutputLayer(self.thetas, self.reg)

    def predict(self, x):

        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x

    def loss(self, x, y):
        score = self.predict(x)
        return self.lastLayer.forward(score, y)

    

    def accuracy(self, x, y):

        score = self.predict(x)
        score = np.argmax(score, axis=1)
        if y.ndim != 1 : y = np.argmax(y, axis=1)
        accuracy = np.sum(score == y) / float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, y):

        loss_W = lambda W: self.loss(x, y)

        grads = {}
        grads['T1'] = numerical_gradient(loss_W, self.params['T1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['T2'] = numerical_gradient(loss_W, self.params['T2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        if self.reg != 0.0:
            pass
            
        return grads

        

    def gradient(self, x, y):

        # forward
        self.loss(x, y)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['T1'], grads['b1'] = self.layers['Affine1'].dT, self.layers['Affine1'].db
        grads['T2'], grads['b2'] = self.layers['Affine2'].dT, self.layers['Affine2'].db
        
        if self.reg != 0.0:
            
            # ========================================== WRITE YOUR CODE ========================================== #
            # Instructions :
            #    Implement the effect of regularization to the gradient
            #    Consider which value should be regularized
            

            grads['T1'] += self.reg*self.params['T1']
            grads['b1'] += self.reg*self.params['b1']
            grads['T2'] += self.reg*self.params['T2']
            grads['b2'] += self.reg*self.params['b2']
            
            
            
            # ===================================================================================================== #

        return grads
    
    
    
    
    
class ThreeLayerNet:

    # ========================================== WRITE YOUR CODE ========================================== #
    # Instructions :
    #     Implement three layers net
    #
    #    __init__() : 
    #        A function that initialize Weight and bias
    #        You should construct a model using the initialized Weight and bias
    #
    #        Model Structure would be like
    #            " Input => Fully Connected => ReLU => Fully Connected => ReLU => Fully Connected => OutputLayer "
    #        Use the classes(Affine, ReLU, OutputLayer) you had defined above
    #        Use hiden_size1, hidden_size2 as variable of the Hidden Layer 
    #    
    #    predict() :
    #        A function that performs forward propagation of Neural network about the Input data(x)
    #    
    #    loss() : 
    #        A function that computes the Loss using the forward propagation results of Neural network with respect to the Input data(x)
    #    
    #    accuracy() :
    #        A function that computes the accuracy using the Output data and True label
    #    
    #
    #    gradient():
    #        A function that performs backward propagation of Neural network using the Input data (x) and True label(y)
        
    

    
    
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, theta_init_std = 0.01, regularization = 0.0):

        # Weight Initialization
        self.params = {}
        self.params['T1'] = theta_init_std * np.random.randn(input_size, hidden_size1)
        self.params['b1'] = np.zeros(hidden_size1)
        self.params['T2'] = theta_init_std * np.random.randn(hidden_size1, hidden_size2) 
        self.params['b2'] = np.zeros(hidden_size2)
        self.params['T3'] = theta_init_std * np.random.randn(hidden_size2, output_size) 
        self.params['b3'] = np.zeros(output_size)

        self.thetas = {}
        self.thetas['T1'] = self.params['T1']
        self.thetas['T2'] = self.params['T2']
        self.thetas['T3'] = self.params['T3']
        
        self.reg = regularization

        self.layers = OrderedDict()
        
        self.layers['Affine1'] = Affine(self.params['T1'], self.params['b1'])
        self.layers['relu'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['T2'], self.params['b2'])
        self.layers['relu1'] = ReLU()
        self.layers['Affine3'] = Affine(self.params['T3'], self.params['b3'])
        
        self.lastLayer = OutputLayer(self.thetas, self.reg)
    
    
    def predict(self, x):

        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x

    def loss(self, x, y):
        score = self.predict(x)
        return self.lastLayer.forward(score, y)

    

    def accuracy(self, x, y):

        score = self.predict(x)
        score = np.argmax(score, axis=1)
        if y.ndim != 1 : y = np.argmax(y, axis=1)
        accuracy = np.sum(score == y) / float(x.shape[0])

        return accuracy      

    def gradient(self, x, y):

        # forward
        self.loss(x, y)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['T1'], grads['b1'] = self.layers['Affine1'].dT, self.layers['Affine1'].db
        grads['T2'], grads['b2'] = self.layers['Affine2'].dT, self.layers['Affine2'].db
        grads['T3'], grads['b3'] = self.layers['Affine3'].dT, self.layers['Affine3'].db
        
        if self.reg != 0.0:
            
            
            grads['T1'] += self.reg*self.params['T1']
            grads['b1'] += self.reg*self.params['b1']
            grads['T2'] += self.reg*self.params['T2']
            grads['b2'] += self.reg*self.params['b2']
            grads['T3'] += self.reg*self.params['T3']
            grads['b3'] += self.reg*self.params['b3']
            

        return grads
    
    
    
    
    
    # ===================================================================================================== #
    
    
    
class simple_CNN(nn.Module):
    def __init__(self, num_class=100):
        super(simple_CNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(6, 16, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        
        self.classifier = nn.Sequential(
            # ========================================== WRITE YOUR CODE ========================================== #
            # Instructions :
            #    Construct a simple CNN model using the pytorch module
            #    Use torch.nn.Sequential to stack the layers
            #
            #    Use 4096, 4096, n(number of classes) Linear layers
            #    Use ReLU activation.
            
            nn.Linear(576, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
            
            # ===================================================================================================== #
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output
    

class deep_CNN(nn.Module):

    # ========================================== WRITE YOUR CODE ========================================== #
    # Instructions :
    #    Construct a deeper CNN model
    #    You have two choices
    #        1. Build a model as you wish
    #        2. Build a model following the instructions below :
    #
    #             Feature extraction :
    #
    #                Use Batch Normalization layer, ReLU activation and Max Pooling(MP) layer 
    #                    and place them appropriately within the convolution layers 
    #                Use (64-64)-MP-(128-128)-MP-(256-256)-MP-(512-512)-MP-(512-512) convolution layers with kernel size 3 and padding 1
    #                Inferring VGG-13 would be helpful
    #         
    #             Classification:
    #
    #                 Use 4096, 4096, n(number of classes) Linear layers
    #
    #                 Use Dropout layer with probability 0.5 and ReLU activation.
    # 
    #    Whatever your choice is, score would be the same as long as the model achieves accuracy higher than 60%
    '''    DO NOT JUST COPY THE CODE FROM THE INTERNET 
            (Inferring them and rewriting it in your own words are fine) '''
    
    
    
    def __init__(self, num_class=100):
        super(deep_CNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_class, bias=True)
        )
    
    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output
    
    
    
    # ===================================================================================================== #



    
