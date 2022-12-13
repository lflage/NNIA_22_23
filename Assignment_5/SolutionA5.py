#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:21:37 2022

@author: benedict
"""

from torch.autograd.functional import jacobian,hessian
from torch import tensor,inverse,matmul
import torch
import numpy as np
#from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm

#######################
#### Newton method ####
#######################

# Defining function
def f(xy):
	return (xy[0]**2-xy[1]**2+4-3*xy[0]*xy[1])

#Defining input tensors
xy = tensor([-0.3,0.3])
#print('xy: ',xy)


def newton(f,x0,step):
    
    xn = x0
    J = jacobian(f,xn)
    H = hessian(f,xn)
    H_inv = inverse(H)
    
    xn1 = xn - matmul(H_inv,J)
    
    J = jacobian(f,xn1)
    step +=1
    print('Point: ',xn1)
    print('Steps: ',step)
    
    if torch.sum(J) != 0:
        newton(f,xn1)
    
    else:
        return xn1


#######################
#### Plotting ####
#######################

def show_plot():
    #fig = plt.figure()
    ax = plt.axes(projection='3d')               
    
    x_surf=np.arange(-0.5, 0.5, 0.01)                
    y_surf=np.arange(-0.5, 0.5, 0.01)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = f(np.array([x_surf,y_surf]))            
    ax.plot_surface(x_surf, y_surf, z_surf, cmap=cm.viridis,alpha=0.3)    # surface plot
    
    x=[-0.3,0]              
    y=[0.3,0]
    z=[4.27,4]
    ax.scatter(x, y, z)                        # scatter plot
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.set_title("Newton's method")
    
    plt.show()
    