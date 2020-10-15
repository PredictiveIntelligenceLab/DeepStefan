# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 20:27:53 2020

@author: sifan
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns
from Stefan_models_tf import Sampler, DataSampler, Stefan1D_inverse_I
import pandas as pd
import os


if __name__ == '__main__':

    def u(z):
        # x = (x, t)
        x = z[:, 0: 1]
        t = z[:, 1: 2]
        u = - 0.5 * x**2  + 2 * x - t - 0.5
        return u

    def s(x):
        t = x[:, 1: 2]
        s = 2 - np.sqrt(3 - 2 * t)
        return s

    def h(x):
        # du / dx (s(t), t) = h(t)   Stefan Neumann Condition
        t = x[:, 1: 2]
        h = np.sqrt(3 - 2 * t) 
        return h

    def g(x):
        # - du /dx (0, t) = g(t)  Neumann Condition
        N = x.shape[0]
        return 2.0 * np.ones((N,1))

    def u_0(x):
        #  Initial Condition
        x = x[:, 0: 1]
        return - 0.5 * x**2 + 2 * x - 0.5

    def S_0(x):
        # Initial Condtion for S(0)
        S_0 = 2 - np.sqrt(3)
        N = x.shape[0]
        return S_0 * np.ones((N, 1))

    # Domain boundaries
    ic_coords = np.array([[0.0, 0.0],
                          [1.0, 0.0]])
    Nc_coords = np.array([[0.0, 0.0],
                          [0.0, 1.0]])
    dom_coords = np.array([[0.0, 0.0],
                           [1.0, 1.0]])

    # Create boundary conditions samplers
    ics_sampler = Sampler(2, ic_coords, lambda x: u_0(x), name='Initial Condition')
    Ncs_sampler = Sampler(2, Nc_coords, lambda x: h(x), name='Stefan Neumann Boundary Condition')  # because t_u_tf

    # Create residual sampler
    res_sampler = Sampler(2, dom_coords, lambda x: u(x), name='Forcing')

    # Define model
    layers_u = [2, 100, 100, 100, 1]
    layers_s = [1, 100, 100, 100, 1]  # or we can map s to (t, s(t))
    
    model = Stefan1D_inverse_I(layers_u, ics_sampler, Ncs_sampler, res_sampler)
    
    # Train the model
    model.train(nIter=40000, batch_size=128)

    # Test data
    nn = 200
    x = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
    t = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]

    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    X_bc_star = np.hstack((np.zeros_like(t), t))
    
    # Exact solutions
    u_star = u(X_star)
    u_bc_star = u(X_bc_star)
    s_star = s(X_star)
   
    # Predictions
    u_pred = model.predict_u(X_star)
    u_bc_pred= model.predict_u(X_bc_star)
    u_x_bc_pred= model.predict_u_x(X_bc_star)
    
    # errors
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_u_bc = np.linalg.norm(u_bc_pred - u_bc_star, 2) / np.linalg.norm(u_bc_star, 2)
    error_u_x_bc = np.linalg.norm(u_x_bc_pred - 2, 2) \
                    / np.linalg.norm(2 * np.ones_like(u_x_bc_pred), 2)
                    
    U_star = griddata(X_star, u_star.flatten(), (X, T), method='cubic')
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    
    u_star_grid = U_star.copy()
    u_pred_grid = U_pred.copy()
    
    for i in range(nn):
        for j in range(nn):
            X_ij = np.array([X[i,j], T[i,j]]).reshape(1,2)
            u_ij = u(X_ij)
            s_ij = s(X_ij)
            if X[i,j] > s_ij:
                U_star[i,j] = np.nan
                U_pred[i,j] = np.nan
                u_star_grid[i,j] = 0
                u_pred_grid[i,j] = 0
                
    print('Relative L2 error_u: {:.2e}'.format(error_u))
    print('Relative L2 error_u_bc: {:.2e}'.format(error_u_bc))
    print('Relative L2 error_u_x_bc: {:.2e}'.format(error_u_x_bc))
    
   
    # Plots for u
    t = np.linspace(0,1, 100)[:, None]
    x = np.zeros_like(t)
    x_star = np.concatenate((x,t), axis=1)
    
    s_star = s(x_star)
    s_pred = model.predict_s(x_star)
    error_s = np.abs(s_star - s_pred)
    
    fig_1 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    
    plt.plot(s_star, t, 'black')
    plt.pcolor(X, T, U_star, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title('Exact $u(x,t)$')
    
    plt.subplot(1, 3, 2)
    plt.pcolor(X, T, U_pred, cmap='jet')
    plt.plot(s_star, t, 'black')
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title('Predicted $u(x,t)$')

    plt.subplot(1, 3, 3)
    plt.pcolor(X, T, np.abs(U_star - U_pred), cmap='jet')
    plt.colorbar(format='%.0e')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title('Absolute Error')
    
    plt.tight_layout()
    plt.show()
    
    
    # Plot for Neumann boundary condition
    X_bc_star = np.hstack((np.zeros_like(t), t))
    X_bc_star = np.hstack((np.zeros_like(t), t))
    u_bc_star = u(X_bc_star)
    
    u_bc_pred= model.predict_u(X_bc_star)
    u_bc_pred= model.predict_u(X_bc_star)
    u_x_bc_pred= model.predict_u_x(X_bc_star)
    
    fig_2 = plt.figure(2, figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(t, u_bc_star, label='Exact') 
    plt.plot(t, u_bc_pred, '--', label='Predicted')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$u(0,t)$')
    plt.title('Boundary Condition')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(t, 2 * np.ones_like(t), label='Exact')
    plt.plot(t, u_x_bc_pred, '--', label='Predicted')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$u_x(0,t)$')
    plt.ylim(1.5,2.5)
    plt.title('Neumann Condition')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    