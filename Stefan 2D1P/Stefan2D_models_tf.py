# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:28:51 2020

@author: sifan
"""

import tensorflow as tf
import numpy as np
import timeit


class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name=None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    def sample(self, N):
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.uniform(0, 1,
                                                                                                  size=(N, self.dim))
        y = self.func(x)
        return x, y


class DataSampler:
    # Initialize the class
    def __init__(self, X, Y, name=None):
        self.X = X
        self.Y = Y
        self.N = self.X.shape[0]

    def sample(self, batch_size):
        idx = np.random.choice(self.N, batch_size, replace=True)
        X_batch = self.X[idx, :]
        Y_batch = self.Y[idx, :]
        return X_batch, Y_batch


class Stefan2D_direct:
    def __init__(self, layers_u, layers_s, ics_sampler, bcs_sampler, res_sampler):
        # Normalization constants
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]
        self.mu_y, self.sigma_y = self.mu_X[1], self.sigma_X[1]
        self.mu_t, self.sigma_t = self.mu_X[2], self.sigma_X[2]

        # Samplers
        self.ics_sampler = ics_sampler
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler

        # Initialize network weights and biases
        self.layers_u = layers_u
        self.weights_u, self.biases_u = self.initialize_NN(layers_u)
        
        # Initialize encoder weights and biases
        self.layers_s = layers_s
        self.weights_s, self.biases_s = self.initialize_NN(layers_s)

        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Define placeholders and computational graph
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))  
        self.y_u_tf = tf.placeholder(tf.float32, shape=(None, 1))  
        self.t_u_tf = tf.placeholder(tf.float32, shape=(None, 1)) 
        self.u_tf = tf.placeholder(tf.float32, shape=(None, 1)) 
        self.s_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        
        self.x_bc3_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_bc3_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_bc3_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc3_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_0_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_0_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_0_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_0_tf = tf.placeholder(tf.float32, shape=(None, 1))  
        self.s_0_tf = tf.placeholder(tf.float32, shape=(None, 1))  

        self.y_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        # Evaluate predictions
        self.s_pred = self.net_s(self.y_u_tf, self.t_u_tf)
        self.s_0_pred = self.net_s(self.y_0_tf, self.t_0_tf)

        self.u_pred = self.net_u(self.x_u_tf, self.y_u_tf, self.t_u_tf)
        self.u_0_pred = self.net_u(self.x_0_tf, self.y_0_tf, self.t_0_tf)

        self.u_bc1_pred = self.net_u(self.x_bc1_tf, self.y_bc1_tf, self.t_bc1_tf)
        self.u_bc2_pred = self.net_u(self.x_bc2_tf, self.y_bc2_tf, self.t_bc2_tf)
        self.u_bc3_pred = self.net_u(self.x_bc3_tf, self.y_bc3_tf, self.t_bc3_tf)
        
        self.S_bc_pred = self.net_u((self.s_pred - self.mu_x) / self.sigma_x,
                                    self.y_u_tf,
                                    self.t_u_tf)
        self.r_Nc_pred = self.net_r_Nc(self.y_Nc_tf, self.t_Nc_tf)
        self.r_u_pred = self.net_r_u(self.x_r_tf, self.y_r_tf, self.t_r_tf)

        # Stefan Boundary loss
        self.loss_Sbc = tf.reduce_mean(tf.square(self.S_bc_pred))
        self.loss_s_0 = tf.reduce_mean(tf.square(self.s_0_pred - (0.5 * (self.y_0_tf * self.sigma_y + self.mu_y) + 0.5))) # s(y, 0) = y/2   + 1/2
        self.loss_SNc = tf.reduce_mean(tf.square(self.r_Nc_pred))

        # Boundary and Initial loss
        self.loss_u_bc1 = tf.reduce_mean(tf.square(self.u_bc1_pred- self.u_bc1_tf))
        self.loss_u_bc2 = tf.reduce_mean(tf.square(self.u_bc2_pred- self.u_bc2_tf))
        self.loss_u_bc3 = tf.reduce_mean(tf.square(self.u_bc3_pred- self.u_bc3_tf))
        self.loss_u_ic = tf.reduce_mean(tf.square(self.u_0_pred - self.u_0_tf))
        
        self.loss_u_bcs = self.loss_u_bc1 + self.loss_u_bc2 + self.loss_u_bc3
        # Stefan loss
        self.loss_Scs = self.loss_Sbc + self.loss_s_0 + self.loss_SNc

        # Residual loss
        self.loss_res = tf.reduce_mean(tf.square(self.r_u_pred))  # u_t - u_xx = 0

        # Total loss
        self.loss = self.loss_u_bcs + self.loss_u_ic + self.loss_Scs + self.loss_res

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        1000, 0.9, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.loss_u_bcs_log = []
        self.loss_u_ic_log = []
        self.loss_res_log = []
        self.saver = tf.train.Saver()
        
        # Estimate the accuracy in the training
        y = np.linspace(0, 1, 100)[:, None]
        t = np.linspace(0, 1, 100)[:, None]
        y, t = np.meshgrid(y, t)

        self.X_star = np.hstack((y.flatten()[:, None], t.flatten()[:, None]))
        self.s_star = self.exact_s(self.X_star)

        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

     # Xavier initialization
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                           dtype=tf.float32)

    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    # Evaluates the forward pass
    def forward_pass_u(self, H):
        num_layers = len(self.layers_u)
        for l in range(0, num_layers - 2):
            W = self.weights_u[l]
            b = self.biases_u[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights_u[-1]
        b = self.biases_u[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H
    
    def forward_pass_s(self, H):
        num_layers = len(self.layers_s)
        for l in range(0, num_layers - 2):
            W = self.weights_s[l]
            b = self.biases_s[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights_s[-1]
        b = self.biases_s[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

     # Forward pass for u
    def net_u(self, x, y, t):
        u = self.forward_pass_u(tf.concat([x, y, t], 1))
        return u

     # Forward pass for s
    def net_s(self, y, t):
        s = self.forward_pass_s(tf.concat([y, t], 1))
        return s

    def exact_s(self, z):
        y = z[:, 0: 1]
        t = z[:, 1: 2]
        s = 0.5 * y + 1.25 * t + 0.5
        return s

    def net_u_x(self, x, y, t):
        u = self.forward_pass_u(tf.concat([x, y, t], 1))
        u_x = tf.gradients(u, x)[0] / self.sigma_x
        u_y = tf.gradients(u, y)[0] / self.sigma_y
        return  u_x, u_y

    # Forward pass for residual
    def net_r_u(self, x, y, t):
        u = self.net_u(x, y, t)
        u_t = tf.gradients(u, t)[0] / self.sigma_t

        u_x = tf.gradients(u, x)[0] / self.sigma_x
        u_y = tf.gradients(u, y)[0] / self.sigma_y

        u_xx = tf.gradients(u_x, x)[0] / self.sigma_x
        u_yy = tf.gradients(u_y, y)[0] / self.sigma_y
        residual = u_t - u_xx - u_yy
        return residual

    def net_r_Nc(self, y, t):
        s = self.net_s(y, t)
        s_y = tf.gradients(s, y)[0] / self.sigma_y
        s_t = tf.gradients(s, t)[0] / self.sigma_t

        # Normalizing s
        s = (s - self.mu_x) / self.sigma_x
        u_x, u_y = self.net_u_x(s, y, t)

        residual = u_x - u_y * s_y + s_t
        return residual

    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def train(self, nIter=10000, batch_size=128):
        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch boundary and data mini-batches
            X_0_batch, u_0_batch = self.fetch_minibatch(self.ics_sampler, batch_size)
            X_bc1_batch, u_bc1_batch = self.fetch_minibatch(self.bcs_sampler[0], batch_size)
            X_bc2_batch, u_bc2_batch = self.fetch_minibatch(self.bcs_sampler[1], batch_size)
            X_bc3_batch, u_bc3_batch = self.fetch_minibatch(self.bcs_sampler[2], batch_size)
            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x_u_tf: X_res_batch[:, 0:1], self.y_u_tf: X_res_batch[:, 1:2],
                       self.t_u_tf: X_res_batch[:, 2:3],
                       self.x_0_tf: X_0_batch[:, 0:1], self.y_0_tf: X_0_batch[:, 1:2],
                       self.t_0_tf: X_0_batch[:, 2:3], self.u_0_tf: u_0_batch,
                       self.x_bc1_tf: X_bc1_batch[:, 0:1], self.y_bc1_tf: X_bc1_batch[:, 1:2],
                       self.t_bc1_tf: X_bc1_batch[:, 2:3], self.u_bc1_tf: u_bc1_batch,
                       self.x_bc2_tf: X_bc2_batch[:, 0:1], self.y_bc2_tf: X_bc2_batch[:, 1:2],
                       self.t_bc2_tf: X_bc2_batch[:, 2:3], self.u_bc2_tf: u_bc2_batch,
                       self.x_bc3_tf: X_bc3_batch[:, 0:1], self.y_bc3_tf: X_bc3_batch[:, 1:2],
                       self.t_bc3_tf: X_bc3_batch[:, 2:3], self.u_bc3_tf: u_bc3_batch,
                       self.y_Nc_tf: X_res_batch[:, 1:2], self.t_Nc_tf: X_res_batch[:, 2:3],
                       self.x_r_tf: X_res_batch[:, 0:1], self.y_r_tf: X_res_batch[:, 1:2],
                       self.t_r_tf: X_res_batch[:, 2:3]}

            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_u_bcs_value, loss_u_ic_value, loss_res_value = self.sess.run(
                    [self.loss_u_bcs, self.loss_u_ic, self.loss_res], tf_dict)
                self.loss_u_bcs_log.append(loss_u_bcs_value)
                self.loss_u_ic_log.append(loss_u_ic_value)
                self.loss_res_log.append(loss_res_value)
                print('It: %d, Loss: %.3e, Loss_bcs: %.3e, Loss_ics: %.3e, Loss_res: %.3e, Time: %.2f' %
                      (it, loss_value, loss_u_bcs_value, loss_u_ic_value, loss_res_value, elapsed))
                start_time = timeit.default_timer()
                
            if it % 100 ==0:
                s_pred = self.predict_s(self.X_star)
                error_s = np.linalg.norm(self.s_star - s_pred, 2) / np.linalg.norm(self.s_star, 2)
                print("Free boundary L2 error: {:.3e}".format(error_s))

    # Predictions for u
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_u_tf: X_star[:, 0:1], self.y_u_tf: X_star[:, 1:2], self.t_u_tf: X_star[:, 2:3]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

    # Predictions for s
    def predict_s(self, X_star):
        X_star = (X_star - self.mu_X[1:3]) / self.sigma_X[1:3]
        tf_dict = {self.y_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]}
        s_star = self.sess.run(self.s_pred, tf_dict)
        return s_star


class Stefan2D_inverse_I:
    def __init__(self, layers_u, layers_s, ics_sampler, ft_sampler, res_sampler):
        # Normalization constants
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]
        self.mu_y, self.sigma_y = self.mu_X[1], self.sigma_X[1]
        self.mu_t, self.sigma_t = self.mu_X[2], self.sigma_X[2]

        # Samplers
        self.ics_sampler = ics_sampler
        self.ft_sampler = ft_sampler
        self.res_sampler = res_sampler

        # Initialize network weights and biases
        self.layers_u = layers_u
        self.weights_u, self.biases_u = self.initialize_NN(layers_u)

        self.layers_s = layers_s
        self.weights_s, self.biases_s = self.initialize_NN(layers_s)

        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Define placeholders and computational graph
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))  # [0, 1]
        self.y_u_tf = tf.placeholder(tf.float32, shape=(None, 1))  # [0, 1]
        self.t_u_tf = tf.placeholder(tf.float32, shape=(None, 1))  # [0, 1]
        self.u_tf = tf.placeholder(tf.float32, shape=(None, 1))  # u(x,t)

        self.x_0_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_0_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_0_tf = tf.placeholder(tf.float32, shape=(None, 1))  # t = 0
        self.u_0_tf = tf.placeholder(tf.float32, shape=(None, 1))  # s(y,0)
        self.s_0_tf = tf.placeholder(tf.float32, shape=(None, 1))  # s(y,0)

        self.x_T_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_T_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_T_tf = tf.placeholder(tf.float32, shape=(None, 1))  # t = 0
        self.u_T_tf = tf.placeholder(tf.float32, shape=(None, 1))  # s(y,0)

        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        # Evaluate predictions
        self.s_pred = self.net_s(self.y_u_tf, self.t_u_tf)
        self.s_0_pred = self.net_s(self.y_0_tf, self.t_0_tf)

        self.u_pred = self.net_u(self.x_u_tf, self.y_u_tf, self.t_u_tf)
        self.u_0_pred = self.net_u(self.x_0_tf, self.y_0_tf, self.t_0_tf)
        self.u_T_pred = self.net_u(self.x_T_tf, self.y_T_tf, self.t_T_tf)

        self.S_bc_pred = self.net_u((self.s_pred - self.mu_x) / self.sigma_x,
                                    self.y_u_tf,
                                    self.t_u_tf)
        self.r_Nc_pred = self.net_r_Nc(self.y_r_tf, self.t_r_tf)
        self.r_u_pred = self.net_r_u(self.x_r_tf, self.y_r_tf, self.t_r_tf)

        # Stefan Boundary loss
        self.loss_s_0 = tf.reduce_mean(tf.square(self.s_0_pred - (0.5 * (self.y_0_tf * self.sigma_y + self.mu_y) + 0.5)))  # s(y, 0) = y/2   + 1/2
        self.loss_Sbc = tf.reduce_mean(tf.square(self.S_bc_pred))
        self.loss_SNc = tf.reduce_mean(tf.square(self.r_Nc_pred))

        # Initial loss
        self.loss_u_0 = tf.reduce_mean(tf.square(self.u_0_pred - self.u_0_tf))

        # Final Time loss
        self.loss_u_T = tf.reduce_mean(tf.square(self.u_T_pred - self.u_T_tf))

        self.loss_u = self.loss_u_0 + self.loss_u_T

        # Stefan loss
        self.loss_Scs = self.loss_s_0 + self.loss_Sbc + self.loss_SNc
        # Neumann condition is important!

        # Residual loss
        self.loss_res = tf.reduce_mean(tf.square(self.r_u_pred))  # u_t - u_xx = 0

        # Total loss
        self.loss = self.loss_u + self.loss_Scs + self.loss_res

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        1000, 0.9, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.loss_u_log = []
        self.loss_res_log = []
        self.saver = tf.train.Saver()

        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)


    # Xavier initialization
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                           dtype=tf.float32)

    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    # Evaluates the forward pass
    def forward_pass_u(self, H):
        num_layers = len(self.layers_u)
        for l in range(0, num_layers - 2):
            W = self.weights_u[l]
            b = self.biases_u[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights_u[-1]
        b = self.biases_u[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

    def forward_pass_s(self, H):
        num_layers = len(self.layers_s)
        for l in range(0, num_layers - 2):
            W = self.weights_s[l]
            b = self.biases_s[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights_s[-1]
        b = self.biases_s[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

    # Forward pass for u
    def net_u(self, x, y, t):
        u = self.forward_pass_u(tf.concat([x, y, t], 1))
        return u

    # Forward pass for s
    def net_s(self, y, t):
        s = self.forward_pass_s(tf.concat([y, t], 1))
        return s

    def net_u_x(self, x, y, t):
        u = self.forward_pass_u(tf.concat([x, y, t], 1))
        u_x = tf.gradients(u, x)[0] / self.sigma_x
        u_y = tf.gradients(u, y)[0] / self.sigma_y
        return u_x, u_y

    # Forward pass for residual
    def net_r_u(self, x, y, t):
        u = self.net_u(x, y, t)
        u_t = tf.gradients(u, t)[0] / self.sigma_t

        u_x = tf.gradients(u, x)[0] / self.sigma_x
        u_y = tf.gradients(u, y)[0] / self.sigma_y

        u_xx = tf.gradients(u_x, x)[0] / self.sigma_x
        u_yy = tf.gradients(u_y, y)[0] / self.sigma_y
        residual = u_t - u_xx - u_yy
        return residual

    def net_r_Nc(self, y, t):
        s = self.net_s(y, t)
        s_y = tf.gradients(s, y)[0] / self.sigma_y
        s_t = tf.gradients(s, t)[0] / self.sigma_t
        
        # Normalizing s
        s = (s - self.mu_x) / self.sigma_x
        
        u_x, u_y = self.net_u_x(s, y, t)

        residual = u_x - tf.multiply(u_y, s_y) + s_t
        return residual


    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def train(self, nIter=10000, batch_size=128):
        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch boundary and data mini-batches
            X_0_batch, u_0_batch = self.fetch_minibatch(self.ics_sampler, batch_size)
            X_T_batch, u_T_batch = self.fetch_minibatch(self.ft_sampler, batch_size)
            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x_u_tf: X_res_batch[:, 0:1], self.y_u_tf: X_res_batch[:, 1:2],
                       self.t_u_tf: X_res_batch[:, 2:3],
                       self.x_0_tf: X_0_batch[:, 0:1], self.y_0_tf: X_0_batch[:, 1:2],
                       self.t_0_tf: X_0_batch[:, 2:3], self.u_0_tf: u_0_batch,
                       self.x_T_tf: X_T_batch[:, 0:1], self.y_T_tf: X_T_batch[:, 1:2],
                       self.t_T_tf: X_T_batch[:, 2:3], self.u_T_tf: u_T_batch,
                       self.x_r_tf: X_res_batch[:, 0:1], self.y_r_tf: X_res_batch[:, 1:2],
                       self.t_r_tf: X_res_batch[:, 2:3]}

            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_u_value, loss_res_value = self.sess.run(
                    [self.loss_u, self.loss_res], tf_dict)
                self.loss_u_log.append(loss_u_value)
                self.loss_res_log.append(loss_res_value)
                print('It: %d, Loss: %.3e, Loss_U: %.3e, Loss_res: %.3e, Time: %.2f' %
                      (it, loss_value, loss_u_value, loss_res_value, elapsed))
                start_time = timeit.default_timer()


    # Predictions for u
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_u_tf: X_star[:, 0:1], self.y_u_tf: X_star[:, 1:2], self.t_u_tf: X_star[:, 2:3]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

    # Predictions for s
    def predict_s(self, X_star):
        X_star = (X_star - self.mu_X[1:3]) / self.sigma_X[1:3]
        tf_dict = {self.y_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]}
        s_star = self.sess.run(self.s_pred, tf_dict)
        return s_star


class Stefan2D_inverse_II:
    def __init__(self, layers_u, layers_s, ics_sampler, res_sampler, data_sampler):
        # Normalization constants
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]
        self.mu_y, self.sigma_y = self.mu_X[1], self.sigma_X[1]
        self.mu_t, self.sigma_t = self.mu_X[2], self.sigma_X[2]

        # Samplers
        self.ics_sampler = ics_sampler
        self.res_sampler = res_sampler
        self.data_sampler = data_sampler

        # Initialize network weights and biases
        self.layers_u = layers_u
        self.weights_u, self.biases_u = self.initialize_NN(layers_u)

        self.layers_s = layers_s
        self.weights_s, self.biases_s = self.initialize_NN(layers_s)

        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Define placeholders and computational graph
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))  # [0, 1]
        self.y_u_tf = tf.placeholder(tf.float32, shape=(None, 1))  # [0, 1]
        self.t_u_tf = tf.placeholder(tf.float32, shape=(None, 1))  # [0, 1]
        self.u_tf = tf.placeholder(tf.float32, shape=(None, 1))  # u(x,t)
        self.s_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_data_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_data_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_data_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_data_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.y_0_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_0_tf = tf.placeholder(tf.float32, shape=(None, 1))  # t = 0
        self.s_0_tf = tf.placeholder(tf.float32, shape=(None, 1))  # s(y,0)

        self.y_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        # Evaluate predictions
        self.s_pred = self.net_s(self.y_u_tf, self.t_u_tf)
        self.u_pred = self.net_u(self.x_u_tf, self.y_u_tf, self.t_u_tf)

        self.u_data_pred = self.net_u(self.x_data_tf, self.y_data_tf, self.t_data_tf)
        self.u_bc_pred = self.net_u((self.s_pred - self.mu_x) / self.sigma_x,
                                    self.y_u_tf,
                                    self.t_u_tf)
        self.r_Nc_pred = self.net_r_Nc(self.y_Nc_tf, self.t_Nc_tf)
        self.r_u_pred = self.net_r_u(self.x_r_tf, self.y_r_tf, self.t_r_tf)

        # Stefan Boundary loss
        self.loss_bc = tf.reduce_mean(tf.square(self.u_bc_pred))
        self.loss_s_0 = tf.reduce_mean(tf.square(self.net_s(self.y_0_tf, self.t_0_tf) -
                                                 (0.5 * (self.y_0_tf * self.sigma_y + self.mu_y) + 0.5))) # s(y, 0) = y/2   + 1/2
        self.loss_Nc = tf.reduce_mean(tf.square(self.r_Nc_pred))

        # Data loss
        self.loss_data = tf.reduce_mean(tf.square(self.u_data_pred - self.u_data_tf))

        # Boundary loss
        self.loss_bcs = self.loss_bc + self.loss_s_0 + self.loss_Nc
        # Neumann condition is important!

        # Residual loss
        self.loss_res = tf.reduce_mean(tf.square(self.r_u_pred))  # u_t - u_xx = 0

        # Total loss
        self.loss = self.loss_bcs + self.loss_res + self.loss_data

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        1000, 0.9, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.loss_bcs_log = []
        self.loss_data_log = []
        self.loss_res_log = []
        self.saver = tf.train.Saver()

        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)


     # Xavier initialization
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                           dtype=tf.float32)

    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    # Evaluates the forward pass
    def forward_pass_u(self, H):
        num_layers = len(self.layers_u)
        for l in range(0, num_layers - 2):
            W = self.weights_u[l]
            b = self.biases_u[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights_u[-1]
        b = self.biases_u[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

    def forward_pass_s(self, H):
        num_layers = len(self.layers_s)
        for l in range(0, num_layers - 2):
            W = self.weights_s[l]
            b = self.biases_s[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights_s[-1]
        b = self.biases_s[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

     # Forward pass for u
    def net_u(self, x, y, t):
        u = self.forward_pass_u(tf.concat([x, y, t], 1))
        return u

    def net_s(self, y, t):
        s = self.forward_pass_s(tf.concat([y, t], 1))
        return s

    def net_u_x(self, x, y, t):
        u = self.forward_pass_u(tf.concat([x, y, t], 1))
        u_x = tf.gradients(u, x)[0] / self.sigma_x
        u_y = tf.gradients(u, y)[0] / self.sigma_y
        return  u_x, u_y

    # Forward pass for residual
    def net_r_u(self, x, y, t):
        u = self.net_u(x, y, t)
        u_t = tf.gradients(u, t)[0] / self.sigma_t

        u_x = tf.gradients(u, x)[0] / self.sigma_x
        u_y = tf.gradients(u, y)[0] / self.sigma_y

        u_xx = tf.gradients(u_x, x)[0] / self.sigma_x
        u_yy = tf.gradients(u_y, y)[0] / self.sigma_y
        residual = u_t - u_xx - u_yy
        return residual

    def net_r_Nc(self, y, t):
        s = self.net_s(y, t)
        s_y = tf.gradients(s, y)[0] / self.sigma_y
        s_t = tf.gradients(s, t)[0] / self.sigma_t

        # Normalizing s
        s = (s - self.mu_x) / self.sigma_x
        u_x, u_y = self.net_u_x(s, y, t)

        residual = u_x - tf.multiply(u_y, s_y) + s_t
        return residual

    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def train(self, nIter=10000, batch_size=128):
        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch boundary and data mini-batches
            X_0_batch, _ = self.fetch_minibatch(self.ics_sampler, batch_size)
            X_data_batch, u_data_batch = self.fetch_minibatch(self.data_sampler, batch_size)

            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x_u_tf: X_res_batch[:, 0:1], self.y_u_tf: X_res_batch[:, 1:2],
                       self.t_u_tf: X_res_batch[:, 2:3],
                       self.y_0_tf: X_0_batch[:, 1:2], self.t_0_tf: X_0_batch[:, 2:3],
                       self.y_Nc_tf: X_res_batch[:, 1:2], self.t_Nc_tf: X_res_batch[:, 2:3],
                       self.x_r_tf: X_res_batch[:, 0:1], self.y_r_tf: X_res_batch[:, 1:2],
                       self.t_r_tf: X_res_batch[:, 2:3],
                       self.x_data_tf: X_data_batch[:, 0:1], self.y_data_tf: X_data_batch[:, 1:2],
                       self.t_data_tf: X_data_batch[:, 2:3], self.u_data_tf: u_data_batch}

            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_bcs_value, loss_data_value, loss_res_value = self.sess.run(
                    [self.loss_bcs, self.loss_data, self.loss_res], tf_dict)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_data_log.append(loss_data_value)
                self.loss_res_log.append(loss_res_value)
                print('It: %d, Loss: %.3e, Loss_bcs: %.3e, Loss_Data: %.3e, Loss_res: %.3e, Time: %.2f' %
                      (it, loss_value, loss_bcs_value, loss_data_value, loss_res_value, elapsed))
                start_time = timeit.default_timer()

    # Predictions for  u
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_u_tf: X_star[:, 0:1], self.y_u_tf: X_star[:, 1:2], self.t_u_tf: X_star[:, 2:3]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

    # Predictions for  s
    def predict_s(self, X_star):
        X_star = (X_star - self.mu_X[1:3]) / self.sigma_X[1:3]
        tf_dict = {self.y_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]}
        s_star = self.sess.run(self.s_pred, tf_dict)
        return s_star









