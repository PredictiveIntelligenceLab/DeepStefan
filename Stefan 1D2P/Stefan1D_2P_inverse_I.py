import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns
from Stefan1D_2p_models_tf import Sampler, DataSampler, Stefan1D_2P_inverse_I
import pandas as pd
import os

if __name__ == '__main__':
    
    # Exact u1
    def u1(x):
        # x = (x, t)
        t = x[:, 1:2] 
        x = x[:, 0:1]

        u1 = 2 * (np.exp((t + 0.5 - x) / 2) - 1)
        return u1

    # Exact u2
    def u2(x):
        # x = (x, t)
        t = x[:, 1:2]
        x = x[:, 0:1]

        u2 = np.exp(t + 0.5 - x) - 1
        return u2

    # Exact s
    def s(x):
        t = x[:, 1:2]
        s = t + 0.5
        return s

    # Exact u
    def u(x):
        return np.where(x[:, 0:1] <= s(x), u1(x), u2(x))

    # condition at the final time 
    def u1_T(x):
        x = x[:, 0:1]
        u1_T = 2 * (np.exp((1 + 0.5 - x) / 2) - 1)
        return u1_T

    def u2_T(x):
        # x = (x, t)

        x = x[:, 0:1]
        u2_T = np.exp(1 + 0.5 - x) - 1
        return u2_T
    
    # Initial condition for s
    def s_0(x):
        z = 0.5
        N = x.shape[0]
        return z * np.ones((N, 1))
    
    # Initial condition for u1
    def u1_0(x):
        x = x[:, 0:1]
        u1_0 = 2 * (np.exp((0.5 - x) / 2) - 1)
        return u1_0

    # Initial condition for u2
    def u2_0(x):
        x = x[:, 0:1]
        u2 = np.exp(0.5 - x) - 1
        return u2
    

    # Domain boundaries
    ics_coords = np.array([[0.0, 0.0],
                           [2.0, 0.0]])
    ft_coords = np.array([[0.0, 1.0],
                           [2.0, 1.0]])
    dom_coords = np.array([[0.0, 0.0],
                           [2.0, 1.0]])

    ic1_sampler = Sampler(2, ics_coords, lambda x: u1_0(x), name='Initial Condition')
    ic2_sampler = Sampler(2, ics_coords, lambda x: u2_0(x), name='Initial Condition')

    ics_sampler = [ic1_sampler, ic2_sampler]

    ft1_sampler = Sampler(2, ft_coords, lambda x: u1_T(x), name='Final Time Condition')
    ft2_sampler = Sampler(2, ft_coords, lambda x: u2_T(x), name='Final Time Condition')

    ft_sampler = [ft1_sampler, ft2_sampler]

    # Create residual sampler
    res_sampler = Sampler(2, dom_coords, lambda x: u(x), name='Forcing')

    # Define model
    layers_u = [2, 100, 100, 100, 2]
    layers_s = [1, 100, 100, 100, 1]  
    model = Stefan1D_2P_inverse_I(layers_u, layers_s, ics_sampler, ft_sampler, res_sampler)

    # Train the model
    model.train(nIter=40000, batch_size=128)

    # Test data
    nn = 100
    x = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
    t = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    u_star = u(X_star)
    s_star = s(X_star)

    # Predictions
    u_pred = model.predict_u(X_star)
    s_pred = model.predict_s(X_star)
    
    # Errors
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_s = np.linalg.norm(s_star - s_pred, 2) / np.linalg.norm(s_star, 2)

    print('Relative L2 error_u: {:.2e}'.format(error_u))
    print('Relative L2 error_s: {:.2e}'.format(error_s))

    # Plot
    U_star = griddata(X_star, u_star.flatten(), (X, T), method='cubic')
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

    t = np.linspace(0, 1, 100)[:, None]
    x = np.zeros_like(t)
    x_star = np.concatenate((x, t), axis=1)

    s_star = s(x_star)
    s_pred = model.predict_s(x_star)
    error_s = np.abs(s_star - s_pred)
    
    # Plot for solution u
    fig_1 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(s_star, t, color='black', linewidth=2)
    plt.pcolor(X, T, U_star, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title('Exact $u(x,t)$')

    plt.subplot(1, 3, 2)
    plt.pcolor(X, T, U_pred, cmap='jet')
    plt.plot(s_pred, t, color='black', linewidth=2)
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






