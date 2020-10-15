# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:51:49 2020

@author: sifan
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:26:40 2020

@author: sifan
"""

import tensorflow as tf
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns
from Stefan2D_models_tf import Sampler, DataSampler, Stefan2D_inverse_I
import pandas as pd
import os


if __name__ == '__main__':
    def u(z):
        # x = (x, y, t)
        x = z[:, 0: 1]
        y = z[:, 1: 2]
        t = z[:, 2: 3]

        u =  np.exp(1.25 * t - x + 0.5 * y + 0.5) - 1
        return u

    def s(z):
        y = z[:, 0: 1]
        t = z[:, 1: 2]
        s = 0.5 * y + 1.25 * t + 0.5
        return s

    def f(z):
        # Initial condition u(x, y, 0) = exp(-x + y/2 + 1/2) -1
        x = z[:, 0: 1]
        y = z[:, 1: 2]
        
        f = np.exp(- x + 0.5 * y + 0.5) - 1
        return f
    
    def h(z):
        # Initial condition u(x, y, T) = exp(1.25  -x + y/2 + 1/2) -1
        x = z[:, 0: 1]
        y = z[:, 1: 2]
        f = np.exp(1.25 - x + 0.5 * y + 0.5) - 1
        return f

    def g1(z):
        # Boundary condition u(x, 0, t) = g1(x, t)
        x = z[:, 0: 1]
        t = z[:, 2: 3]

        g1 = np.exp(1.25 * t - x + 0.5) - 1
        return g1
    
    def g2(z):
        # Boundary condition u(0, y, t) = g2(y, t)
        y = z[:, 1: 2]
        t = z[:, 2: 3]
    
        g2 = np.exp(1.25 * t + 0.5 * y + 0.5) - 1
        return g2

    def z(x):
        N = x.shape[0]
        return np.zeros((N, 1))
    
     # Domain boundaries
    ic_coords = np.array([[0.0, 0.0, 0.0],
                          [2.25, 1.0, 0.0]])
    ft_coords = np.array([[0.0, 0.0, 1.0],
                          [2.25, 1.0, 1.0]])
    
    dom_coords = np.array([[0.0, 0.0, 0.0],
                           [2.25, 1.0, 1.0]])

    # Create Initial conditions samplers
    ics_sampler = Sampler(3, ic_coords, lambda x: u(x), name='Initial Condition')

    ft_sampler = Sampler(3, ft_coords, lambda x: u(x))
    # Create residual sampler
    res_sampler = Sampler(3, dom_coords, lambda x: u(x), name='Forcing')
    
    
    # Define model
    layers_u = [3, 100, 100, 100, 1]
    layers_s = [2, 100, 100, 100, 1]  # or we can map s to (t, s(t))
    model = Stefan2D_inverse_I(layers_u, layers_s, ics_sampler, ft_sampler, res_sampler)
    
    # Train the model
    model.train(nIter=40000, batch_size=128)
    
     # Test data
    nn = 200
    y = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
    t = np.linspace(dom_coords[0, 2], dom_coords[1, 2], nn)[:, None]
    y, t = np.meshgrid(y, t)
    
    X_star = np.hstack((y.flatten()[:, None], t.flatten()[:, None]))
    
    s_star = s(X_star)

    # Predictions
    s_pred = model.predict_s(X_star)
    
    error_s = np.linalg.norm(s_star - s_pred, 2) / np.linalg.norm(s_star, 2)
    print('The relative errror is: {:4e}'.format(error_s))
    

    # Plot for s
    nn = 200
    y = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
    t = np.linspace(dom_coords[0, 2], dom_coords[1, 2], nn)[:, None]
    y, t = np.meshgrid(y, t)

    X_star = np.hstack((y.flatten()[:, None], t.flatten()[:, None]))
    
    s_star = s(X_star)
    s_pred = model.predict_s(X_star)
    
    S_pred = griddata(X_star, s_pred.flatten(), (y, t), method='cubic')
    S_star = griddata(X_star, s_star.flatten(), (y, t), method='cubic')
        
    fig_1 = plt.figure(5, figsize=(18, 5))
    ax = fig_1.add_subplot(1, 3, 1, projection='3d')
    ax.plot_surface(y, t, S_star)
    ax.set_xlabel('y')
    ax.set_ylabel('t')
    ax.set_zlabel('s(y,t)')
    ax.set_title('Exact')
    
    ax = fig_1.add_subplot(1, 3, 2, projection='3d')
    ax.plot_surface(y, t, S_pred)
    ax.set_xlabel('y')
    ax.set_ylabel('t')
    ax.set_zlabel('s(y,t)')
    ax.set_title('Predicted')
    
    ax = fig_1.add_subplot(1, 3, 3, projection='3d')
    ax.plot_surface(y, t, np.abs(S_star - S_pred))
    ax.set_xlabel('y')
    ax.set_ylabel('t')
    ax.set_zlabel('s(y,t)')
    ax.set_title('Absolute Error')
    
    plt.tight_layout()
    plt.show()


    # Plot for u
    T_list = [0.2, 0.4, 0.6, 0.8]
    nn = 200
    x = np.linspace(0, 2.25, nn)[:, None]
    y = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
    t = np.linspace(dom_coords[0, 2], dom_coords[1, 2], nn)[:, None]

    for T in T_list:
        X, Y = np.meshgrid(x, y)
        T_star = T * np.ones_like(X)
        X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], T_star.flatten()[:, None]))
        
        u_star = u(X_star)
        u_pred = model.predict_u(X_star)
        
        X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
        
        U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
        U_pred = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
        
        for i in range(nn):
            for j in range(nn):
                X_ij = np.array([X[i,j], Y[i,j], T]).reshape(1,3)
                u_ij = u(X_ij)
                s_ij = s(np.array([Y[i,j], T]).reshape(1,2))
                if X[i,j] > s_ij:
                    U_star[i,j] = np.nan
                    U_pred[i,j] = np.nan
    
        np.savetxt('pred_{}'.format(str(T)), U_pred, delimiter=',')
        np.savetxt('exact_{}'.format(str(T)), U_star, delimiter=',')
        
        
    for T in T_list:
        U_pred = np.loadtxt('pred_{}'.format(str(T)), delimiter=',')
        U_star = np.loadtxt('exact_{}'.format(str(T)), delimiter=',')
        
        fig = plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.pcolor(X, Y, U_star, cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title('Exact $u(x, y, {})$'.format(T))
        
        plt.subplot(1, 3, 2)
        plt.pcolor(X, Y, U_pred, cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title('Predicted  $u(x, y, {})$'.format(T))
    
        plt.subplot(1, 3, 3)
        plt.pcolor(X, Y, np.abs(U_star - U_pred), cmap='jet')
        plt.colorbar(format='%.0e')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title('Absolute Error')
        
        plt.tight_layout()
        plt.show()

    
  
    
    
    
 