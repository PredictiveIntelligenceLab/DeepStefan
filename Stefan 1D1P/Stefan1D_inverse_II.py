import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns
from Stefan_models_tf import Sampler, DataSampler, Stefan1D_inverse_II
import pandas as pd
import os

if __name__ == '__main__':

    def u(z):
        # x = (x, t)
        # u(x) = e^{t -x} - 1
        x = z[:, 0: 1]
        t = z[:, 1: 2]
        u = - 0.5 * x**2  + 2 * x - t - 0.5
        return u

    def s(x):
        t = x[:, 1: 2]
        s = 2 - np.sqrt(3 - 2 * t)
        return s

    def g(x):
        t = t = x[:, 1: 2]
        g = np.sqrt(3 - 2 * t)

        return g

    def z(x):
        z = 2 - np.sqrt(3)
        N = x.shape[0]
        return z * np.ones((N, 1))


    # Domain boundaries
    bc_coords = np.array([[0.0, 0.0],
                          [0.0, 0.0]])
    Nc_coords = np.array([[0.0, 0.0],
                          [0.0, 1.0]])
    dom_coords = np.array([[0.0, 0.0],
                           [1.0, 1.0]])
    

    # Create boundary conditions samplers
    bcs_sampler = Sampler(2, bc_coords, lambda x: z(x), name='Boundary Condition')
    Ncs_sampler = Sampler(2, dom_coords, lambda x: g(x), name='Neumann Boundary Condition') # because t_u_tf

    # Create residual sampler
    res_sampler = Sampler(2, dom_coords, lambda x: u(x), name='Forcing')

    # Construct total data set
    data_sampler = Sampler(2, dom_coords, lambda x: u(x), name='Forcing')
    data_X, data_u = data_sampler.sample(10**5) # Sample enough points
    
    # Select data points in side the physcial domain
    mask = data_X[:,0:1] < s(data_X)
    data_X = data_X[mask[:,0]]
    data_u = data_u[mask[:,0]]

    num = 10
    data_X, data_u = DataSampler(data_X, data_u).sample(num)
    data_sampler = DataSampler(data_X, data_u)

    # Define model
    layers_u = [2, 100, 100, 100, 1]
    layers_s = [1, 100, 100, 100, 1]  # or we can map s to (t, s(t))
    model = Stefan1D_inverse_II(layers_u, layers_s, bcs_sampler, Ncs_sampler, res_sampler, data_sampler)
    
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
    s_star = s(X_star)
    u_bc_star = u(X_bc_star)
    
    # Predictions
    u_pred = model.predict_u(X_star)
    s_pred = model.predict_s(X_star)
    u_bc_pred = model.predict_u(X_bc_star)
    u_x_bc_pred = model.predict_u_x(X_bc_star)
    
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
                
    # Errors
    error_u = np.linalg.norm(u_star_grid - u_pred_grid, 2) / np.linalg.norm(u_star_grid, 2)
    error_s = np.linalg.norm(s_star - s_pred, 2) / np.linalg.norm(s_star, 2)
    
    print('Relative L2 error_u: {:.2e}'.format(error_u))
    print('Relative L2 error_s: {:.2e}'.format(error_s))

    # Plot
    t = np.linspace(0,1, nn)[:, None]
    x = np.zeros_like(t)
    x_star = np.concatenate((x,t), axis=1)
    
    s_star = s(x_star)
    s_pred = model.predict_s(x_star)
    error_s = np.abs(s_star - s_pred)
    
    # Plot for solution u
    fig_1 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(s_star, t)
    plt.plot(data_X[:,0:1], data_X[:,1:2], 'x', color='black')
    plt.pcolor(X, T, U_star, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title('Exact $u(x,t)$')
    
    plt.subplot(1, 3, 2)
    plt.pcolor(X, T, U_pred, cmap='jet')
    plt.plot(s_pred, t)
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
    
    
    # Plot for solution s
    
    t = np.linspace(0,1, nn)[:, None]
    x = np.zeros_like(t)
    x_star = np.concatenate((x,t), axis=1)
    
    s_star = s(x_star)
    s_pred = model.predict_s(x_star)
    
    fig_2 = plt.figure(2, figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, s_star, label='Exact')
    plt.plot(t, s_pred, '--', label='Predicted')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$s(t)$')
    plt.title('Moving Boundary')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t, error_s)
    plt.xlabel(r'$t$')
    plt.ylabel(r'Point-wise Error')
    plt.title('Absolute Error')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    
 
    