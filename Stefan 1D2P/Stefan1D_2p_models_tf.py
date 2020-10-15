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
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.uniform(0, 1, size=(N, self.dim))
        y = self.func(x)
        return x, y

class DataSampler:
    # Initialize the class
    def __init__(self, X, Y, name = None):
        self.X = X
        self.Y = Y
        self.N = self.X.shape[0]

    def sample(self, batch_size):
        idx = np.random.choice(self.N, batch_size, replace=True)
        X_batch = self.X[idx, :]
        Y_batch = self.Y[idx, :]
        return X_batch, Y_batch

class Stefan1D_2P_direct:
    def __init__(self, layers_u, layers_s, bcs_sampler, ics_sampler, res_sampler):
        # Normalization constants
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]
        self.mu_t, self.sigma_t = self.mu_X[1], self.sigma_X[1]

        # Samplers
        self.bcs_sampler = bcs_sampler
        self.ics_sampler = ics_sampler
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
        self.t_u_tf = tf.placeholder(tf.float32, shape=(None, 1))  # [0, 1]

        self.t_0_tf = tf.placeholder(tf.float32, shape=(None, 1))  # t = 0
        self.s_0_tf = tf.placeholder(tf.float32, shape=(None, 1))  # s(0)

        self.x_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_ic1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_ic1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_ic1_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_ic2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_ic2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_ic2_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        # Evaluate predictions
        self.s_pred = self.net_s(self.t_u_tf)
        self.u1_pred, self.u2_pred = self.net_u1u2(self.x_r_tf, self.t_r_tf)
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)

        self.u1_0_pred,_ = self.net_u1u2(self.x_ic1_tf, self.t_ic1_tf)
        _, self.u2_0_pred = self.net_u1u2(self.x_ic2_tf, self.t_ic2_tf)

        self.u1_bc_pred, _ = self.net_u1u2(self.x_bc1_tf, self.t_bc1_tf)
        _, self.u2_bc_pred = self.net_u1u2(self.x_bc2_tf, self.t_bc2_tf)

        self.s_bc1_pred, self.s_bc2_pred = self.net_u1u2((self.net_s(self.t_r_tf) - self.mu_x) / self.sigma_x, self.t_r_tf)

        self.r_u1_pred, self.r_u2_pred = self.net_r_u1u2(self.x_r_tf, self.t_r_tf)

        self.r_Nc_pred = self.net_r_Nc(self.t_r_tf)

        # Boundary loss
        self.loss_u1_bc = tf.reduce_mean(tf.square(self.u1_bc_pred - self.u_bc1_tf))
        self.loss_u2_bc = tf.reduce_mean(tf.square(self.u2_bc_pred - self.u_bc2_tf))
        self.loss_u_bcs = self.loss_u1_bc + self.loss_u2_bc

        # Initial Loss
        self.loss_u1_ic = tf.reduce_mean(tf.square(self.u1_0_pred - self.u_ic1_tf))
        self.loss_u2_ic = tf.reduce_mean(tf.square(self.u2_0_pred - self.u_ic2_tf))
        self.loss_u_ics = self.loss_u1_ic + self.loss_u2_ic

        # Stefan loss
        self.loss_Sbc1 = tf.reduce_mean(tf.square(self.s_bc1_pred))  # u1(s(t),t) = u2(s(t), t) = 0
        self.loss_Sbc2 = tf.reduce_mean(tf.square(self.s_bc2_pred))  # u1(s(t),t) = u2(s(t), t) = 0
        self.loss_s_0 = tf.reduce_mean(tf.square(self.net_s(self.t_ic1_tf) - 0.5))  # s(0) = 0.5
        self.loss_SNc = tf.reduce_mean(tf.square(self.r_Nc_pred))   # Neumann Condition

        self.loss_Scs = self.loss_Sbc1 + self.loss_Sbc2 + self.loss_s_0 + self.loss_SNc

        # Residual loss
        self.loss_res_u1 = tf.reduce_mean(tf.square(self.r_u1_pred))
        self.loss_res_u2 = tf.reduce_mean(tf.square(self.r_u2_pred))
        self.loss_res = self.loss_res_u1 + self.loss_res_u2

        # Total loss
        self.loss = self.loss_res + self.loss_u_ics + self.loss_u_bcs + self.loss_Scs

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        1000, 0.9, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.loss_bcs_log = []
        self.loss_ics_log = []
        self.loss_Scs_log = []
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
    def forward_pass(self, H, weights, biases):
        num_layers = len(weights)
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

    # Forward pass for u1, u2
    def net_u1u2(self, x, t):
        u = self.forward_pass(tf.concat([x, t], 1), self.weights_u, self.biases_u)
        u1 = u[:,0:1]
        u2 = u[:,1:2]
        return u1, u2

    def net_u1u2_x(self, x, t):
        u1, u2 = self.net_u1u2(x, t)
        u1_x = tf.gradients(u1, x)[0] / self.sigma_x
        u2_x = tf.gradients(u2, x)[0] / self.sigma_x
        
        return u1_x, u2_x

    # Forward pass for u
    def net_s(self, t):
        s = self.forward_pass(t, self.weights_s, self.biases_s)
        return s

    # Forward pass for u
    def net_u(self, x, t):
        u1, u2 = self.net_u1u2(x, t)
        s = self.net_s(t)  # consider normalization

        # demoralizing x coordinates
        x_hat = x * self.sigma_x + self.mu_x
        # consider s = x_hat
        u = tf.multiply(u1, 0.5 * (tf.sign(s - x_hat) + 1)) + tf.multiply(u2, 0.5 * (tf.sign(x_hat - s) + 1))
        return u

    # Forward pass for residual
    def net_r_u1u2(self, x, t):
        u1, u2 = self.net_u1u2(x, t)
        
        u1_t = tf.gradients(u1, t)[0] / self.sigma_t
        u1_x = tf.gradients(u1, x)[0] / self.sigma_x
        u1_xx = tf.gradients(u1_x, x)[0] / self.sigma_x
        r_u1 = u1_t - 2 * u1_xx

        u2_t = tf.gradients(u2, t)[0] / self.sigma_t
        u2_x = tf.gradients(u2, x)[0] / self.sigma_x
        u2_xx = tf.gradients(u2_x, x)[0] / self.sigma_x
        r_u2 = u2_t - u2_xx
        
        return r_u1, r_u2

    def net_r_Nc(self, t):
        s = self.net_s(t)
        s_t = tf.gradients(s, t)[0] / self.sigma_t

        # Normalizing s
        s = (s - self.mu_x) / self.sigma_x

        u1_x, u2_x = self.net_u1u2_x(s, t)

        residual = s_t - u2_x + 2 * u1_x
        return residual

    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def train(self, nIter=10000, batch_size=128):
        start_time = timeit.default_timer()

        for it in range(nIter):
            # Fetch boundary and data mini-batches
            X_bc1_batch, u1_bc_batch = self.fetch_minibatch(self.bcs_sampler[0], batch_size)
            X_bc2_batch, u2_bc_batch = self.fetch_minibatch(self.bcs_sampler[1], batch_size)
            X_ic1_batch, u1_ic_batch = self.fetch_minibatch(self.ics_sampler[0], batch_size)
            X_ic2_batch, u2_ic_batch = self.fetch_minibatch(self.ics_sampler[1], batch_size)

            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x_u_tf: X_res_batch[:, 0:1], self.t_u_tf: X_res_batch[:, 1:2],
                       self.x_bc1_tf: X_bc1_batch[:, 0:1], self.t_bc1_tf: X_bc1_batch[:, 1:2],
                       self.u_bc1_tf: u1_bc_batch,
                       self.x_bc2_tf: X_bc2_batch[:, 0:1], self.t_bc2_tf: X_bc2_batch[:, 1:2],
                       self.u_bc2_tf: u2_bc_batch,
                       self.x_ic1_tf: X_ic1_batch[:, 0:1], self.t_ic1_tf: X_ic1_batch[:, 1:2],
                       self.u_ic1_tf: u1_ic_batch,
                       self.x_ic2_tf: X_ic2_batch[:, 0:1], self.t_ic2_tf: X_ic2_batch[:, 1:2],
                       self.u_ic2_tf: u2_ic_batch,
                       self.x_r_tf: X_res_batch[:, 0:1], self.t_r_tf: X_res_batch[:, 1:2]}

            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_bcs_value, loss_ics_value, loss_res_value = self.sess.run(
                    [self.loss_u_bcs, self.loss_u_ics, self.loss_res], tf_dict)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_ics_log.append(loss_ics_value)
                self.loss_res_log.append(loss_res_value)
                print('It: %d, Loss: %.3e, Loss_bcs: %.3e, Loss_ics: %.3e, Loss_res: %.3e, Time: %.2f' %
                      (it, loss_value, loss_bcs_value, loss_ics_value, loss_res_value, elapsed))
                start_time = timeit.default_timer()

    # Predictions for u
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

    # Predictions for s
    def predict_s(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_u_tf: X_star[:, 1:2]}
        s_star = self.sess.run(self.s_pred, tf_dict)
        return s_star


class Stefan1D_2P_inverse_I:
    def __init__(self, layers_u, layers_s, ics_sampler, ft_sampler, res_sampler):
        # Normalization constants
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]
        self.mu_t, self.sigma_t = self.mu_X[1], self.sigma_X[1]

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
        self.t_u_tf = tf.placeholder(tf.float32, shape=(None, 1))  # [0, 1]

        self.t_0_tf = tf.placeholder(tf.float32, shape=(None, 1))  # t = 0
        self.s_0_tf = tf.placeholder(tf.float32, shape=(None, 1))  # s(0)

        self.x1_ic_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t1_ic_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u1_ic_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x2_ic_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t2_ic_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u2_ic_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x1_T_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t1_T_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u1_T_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x2_T_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t2_T_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u2_T_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        # Evaluate predictions
        self.s_pred = self.net_s(self.t_u_tf)
        self.u1_pred, self.u2_pred = self.net_u1u2(self.x_r_tf, self.t_r_tf)
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)

        self.u1_0_pred, _ = self.net_u1u2(self.x1_ic_tf, self.t1_ic_tf)
        _, self.u2_0_pred = self.net_u1u2(self.x2_ic_tf, self.t2_ic_tf)

        self.u1_T_pred, _ = self.net_u1u2(self.x1_T_tf, self.t1_T_tf)
        _, self.u2_T_pred = self.net_u1u2(self.x2_T_tf, self.t2_T_tf)

        self.s_bc1_pred, self.s_bc2_pred = self.net_u1u2((self.net_s(self.t_r_tf) - self.mu_x) / self.sigma_x, self.t_r_tf)
         
        self.r_u1_pred, self.r_u2_pred = self.net_r_u1u2(self.x_r_tf, self.t_r_tf)

        self.r_Nc_pred = self.net_r_Nc(self.t_r_tf)

        # Boundary loss
        self.loss_u1_T = tf.reduce_mean(tf.square(self.u1_T_pred - self.u1_T_tf))
        self.loss_u2_T = tf.reduce_mean(tf.square(self.u2_T_pred - self.u2_T_tf))
        self.loss_u_T = self.loss_u1_T + self.loss_u2_T

        # Initial Loss
        self.loss_u1_ic = tf.reduce_mean(tf.square(self.u1_0_pred - self.u1_ic_tf))
        self.loss_u2_ic = tf.reduce_mean(tf.square(self.u2_0_pred - self.u2_ic_tf))
        self.loss_u_ics = self.loss_u1_ic + self.loss_u2_ic

        # Stefan loss
        self.loss_Sbc1 = tf.reduce_mean(tf.square(self.s_bc1_pred))  # u1(s(t),t) = u2(s(t), t) = 0
        self.loss_Sbc2 = tf.reduce_mean(tf.square(self.s_bc2_pred))  # u1(s(t),t) = u2(s(t), t) = 0
        self.loss_s_0 = tf.reduce_mean(tf.square(self.net_s(self.t1_ic_tf) - 0.5))  # s(0) = 0.5
        self.loss_SNc = tf.reduce_mean(tf.square(self.r_Nc_pred))   # Neumann Condition

        self.loss_Scs = self.loss_Sbc1 + self.loss_Sbc2 + self.loss_s_0 + self.loss_SNc

        # Residual loss
        self.loss_res_u1 = tf.reduce_mean(tf.square(self.r_u1_pred))
        self.loss_res_u2 = tf.reduce_mean(tf.square(self.r_u2_pred))
        self.loss_res = self.loss_res_u1 + self.loss_res_u2

        # Total loss
        self.loss_u = self.loss_u_ics + self.loss_u_T
        self.loss = self.loss_res + self.loss_u + self.loss_Scs

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        1000, 0.9, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.loss_u_log = []
        self.loss_Scs_log = []
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
    def forward_pass(self, H, weights, biases):
        num_layers = len(weights)
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

     # Forward pass for u1, u2
    def net_u1u2(self, x, t):
        u = self.forward_pass(tf.concat([x, t], 1), self.weights_u, self.biases_u)
        u1 = u[:,0:1]
        u2 = u[:,1:2]
        return u1, u2

    def net_u1u2_x(self, x, t):
        u1, u2 = self.net_u1u2(x, t)
        u1_x = tf.gradients(u1, x)[0] / self.sigma_x
        u2_x = tf.gradients(u2, x)[0] / self.sigma_x
        
        return u1_x, u2_x

     # Forward pass for s
    def net_s(self, t):
        s = self.forward_pass(t, self.weights_s, self.biases_s)
        return s

     # Forward pass for u
    def net_u(self, x, t):
        u1, u2 = self.net_u1u2(x, t)
        s = self.net_s(t)  # consider normalization

        # demoralizing x coordinates
        x_hat = x * self.sigma_x + self.mu_x
        # consider s = x_hat
        u = tf.multiply(u1, 0.5 * (tf.sign(s - x_hat) + 1)) + tf.multiply(u2, 0.5 * (tf.sign(x_hat - s) + 1))
        return u

    # Forward pass for residual
    def net_r_u1u2(self, x, t):
        u1, u2 = self.net_u1u2(x, t)
        
        u1_t = tf.gradients(u1, t)[0] / self.sigma_t
        u1_x = tf.gradients(u1, x)[0] / self.sigma_x
        u1_xx = tf.gradients(u1_x, x)[0] / self.sigma_x
        r_u1 = u1_t - 2 * u1_xx

        u2_t = tf.gradients(u2, t)[0] / self.sigma_t
        u2_x = tf.gradients(u2, x)[0] / self.sigma_x
        u2_xx = tf.gradients(u2_x, x)[0] / self.sigma_x
        r_u2 = u2_t - u2_xx
        
        return r_u1, r_u2

    def net_r_Nc(self, t):
        s = self.net_s(t)
        s_t = tf.gradients(s, t)[0] / self.sigma_t

        # Normalizing s
        s = (s - self.mu_x) / self.sigma_x
        
        u1_x, u2_x = self.net_u1u2_x(s, t)
        residual = s_t - u2_x + 2 * u1_x
        return residual

    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def train(self, nIter=10000, batch_size=128):
        start_time = timeit.default_timer()

        for it in range(nIter):
            # Fetch boundary and data mini-batches
            X1_ic_batch, u1_ic_batch = self.fetch_minibatch(self.ics_sampler[0], batch_size)
            X2_ic_batch, u2_ic_batch = self.fetch_minibatch(self.ics_sampler[1], batch_size)
            X1_T_batch, u1_T_batch = self.fetch_minibatch(self.ft_sampler[0], batch_size)
            X2_T_batch, u2_T_batch = self.fetch_minibatch(self.ft_sampler[1], batch_size)
            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x_u_tf: X_res_batch[:, 0:1], self.t_u_tf: X_res_batch[:, 1:2],
                       self.x1_ic_tf: X1_ic_batch[:, 0:1], self.t1_ic_tf: X1_ic_batch[:, 1:2],
                       self.u1_ic_tf: u1_ic_batch,
                       self.x2_ic_tf: X2_ic_batch[:, 0:1], self.t2_ic_tf: X2_ic_batch[:, 1:2],
                       self.u2_ic_tf: u2_ic_batch,
                       self.x1_T_tf: X1_T_batch[:, 0:1], self.t1_T_tf: X1_T_batch[:, 1:2],
                       self.u1_T_tf: u1_T_batch,
                       self.x2_T_tf: X2_T_batch[:, 0:1], self.t2_T_tf: X2_T_batch[:, 1:2],
                       self.u2_T_tf: u2_T_batch,
                       self.x_r_tf: X_res_batch[:, 0:1], self.t_r_tf: X_res_batch[:, 1:2]}

            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_u_value, loss_Scs_value, loss_res_value = self.sess.run(
                    [self.loss_u, self.loss_Scs, self.loss_res], tf_dict)
                self.loss_u_log.append(loss_u_value)
                self.loss_Scs_log.append(loss_Scs_value)
                self.loss_res_log.append(loss_res_value)
                print('It: %d, Loss: %.3e, Loss_u: %.3e, Loss_Scs: %.3e, Loss_res: %.3e, Time: %.2f' %
                      (it, loss_value, loss_u_value, loss_Scs_value, loss_res_value, elapsed))
                start_time = timeit.default_timer()

    # Predictions for u
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

    # Predictions for s
    def predict_s(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_u_tf: X_star[:, 1:2]}
        s_star = self.sess.run(self.s_pred, tf_dict)
        return s_star

class Stefan1D_2P_inverse_II:
    def __init__(self, layers_u, layers_s, bcs_sampler, Sbc_sampler, SNc_sampler, res_sampler, data_sampler):
        # Normalization constants
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]
        self.mu_t, self.sigma_t = self.mu_X[1], self.sigma_X[1]

        # Samplers
        self.bcs_sampler = bcs_sampler
        self.Sbc_sampler = Sbc_sampler
        self.SNc_sampler = SNc_sampler
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
        self.t_u_tf = tf.placeholder(tf.float32, shape=(None, 1))  # [0, 1]

        self.x_data_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_data_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_data_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_0_tf = tf.placeholder(tf.float32, shape=(None, 1))  # t = 0
        self.s_0_tf = tf.placeholder(tf.float32, shape=(None, 1))  # s(0)
        
        self.t_bc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.s_bc_tf = tf.placeholder(tf.float32, shape=(None, 1))  # s(0)

        self.x_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_r_below_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_r_below_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_r_above_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_r_above_tf = tf.placeholder(tf.float32, shape=(None, 1))

        # Evaluate predictions
        self.s_pred = self.net_s(self.t_u_tf)
        self.u1_pred, self.u2_pred = self.net_u1u2(self.x_r_tf, self.t_r_tf)
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)

        self.s_bc1_pred, self.s_bc2_pred = self.net_u1u2((self.net_s(self.t_bc_tf) - self.mu_x) / self.sigma_x, self.t_bc_tf)

        self.r_u1_pred, self.r_u2_pred = self.net_r_u1u2(self.x_r_tf, self.t_r_tf)

        self.r_Nc_pred = self.net_r_Nc(self.t_Nc_tf)

        self.u_data_pred = self.net_u(self.x_data_tf, self.t_data_tf)

        # Stefan Boundary loss
        self.loss_bc1 = tf.reduce_mean(tf.square(self.s_bc1_pred))   # u1(s(t),t) = u2(s(t), t) = 0
        self.loss_bc2 = tf.reduce_mean(tf.square(self.s_bc2_pred))  # u1(s(t),t) = u2(s(t), t) = 0
        self.loss_s_0 = tf.reduce_mean(tf.square(self.net_s(self.t_0_tf) - self.s_0_tf))  # s(0) = 0.5
        self.loss_Nc = tf.reduce_mean(tf.square(self.r_Nc_pred))                # Neumann Condition

        # Data loss
        self.loss_data = tf.reduce_mean(tf.square(self.u_data_pred - self.u_data_tf))

        # Boundary loss
        self.loss_bcs = self.loss_bc1 + self.loss_bc2 + self.loss_s_0 + self.loss_Nc

        # Residual loss
        self.loss_res_u1 = tf.reduce_mean(tf.square(self.r_u1_pred))
        self.loss_res_u2 = tf.reduce_mean(tf.square(self.r_u2_pred))
        self.loss_res = self.loss_res_u1 + self.loss_res_u2

        # Total loss
        self.loss = self.loss_res + self.loss_data + self.loss_bcs

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
    def forward_pass(self, H, weights, biases):
        num_layers = len(weights)
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

    # Forward pass for u1, u2
    def net_u1u2(self, x, t):
        u = self.forward_pass(tf.concat([x, t], 1), self.weights_u, self.biases_u)
        u1 = u[:,0:1]
        u2 = u[:,1:2]
        return u1, u2

    def net_u1u2_x(self, x, t):
        u1, u2 = self.net_u1u2(x, t)
        u1_x = tf.gradients(u1, x)[0] / self.sigma_x
        u2_x = tf.gradients(u2, x)[0] / self.sigma_x
        
        return u1_x, u2_x

    # Forward pass for s
    def net_s(self, t):
        s = self.forward_pass(t, self.weights_s, self.biases_s)
        return s

    # Forward pass for u
    def net_u(self, x, t):
        u1, u2 = self.net_u1u2(x, t)
        s = self.net_s(t)  # consider normalization

        # demoralizing x coordinates
        x_hat = x * self.sigma_x + self.mu_x
        # consider s = x_hat
        u = tf.multiply(u1, 0.5 * (tf.sign(s - x_hat) + 1)) + tf.multiply(u2, 0.5 * (tf.sign(x_hat - s) + 1))
        return u

    # Forward pass for residual
    def net_r_u1u2(self, x, t):
        u1, u2 = self.net_u1u2(x, t)
        
        u1_t = tf.gradients(u1, t)[0] / self.sigma_t
        u1_x = tf.gradients(u1, x)[0] / self.sigma_x
        u1_xx = tf.gradients(u1_x, x)[0] / self.sigma_x
        r_u1 = u1_t - 2 * u1_xx

        u2_t = tf.gradients(u2, t)[0] / self.sigma_t
        u2_x = tf.gradients(u2, x)[0] / self.sigma_x
        u2_xx = tf.gradients(u2_x, x)[0] / self.sigma_x
        r_u2 = u2_t - u2_xx
        
        return r_u1, r_u2
    
    def net_r_Nc(self, t):
        s = self.net_s(t)
        s_t = tf.gradients(s, t)[0] / self.sigma_t
        
        # Normalizing s
        s = (s - self.mu_x) / self.sigma_x

        u1_x, u2_x = self.net_u1u2_x(s, t)

        residual = s_t - u2_x + 2 * u1_x
        return residual

    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def split_minibatch(self, X_batch, s):

        # denormalizing minibatches
        X_batch_original = X_batch * self.sigma_X + self.mu_X

        mask_above = (X_batch_original[:, 0:1] >= s)
        mask_below = (X_batch_original[:, 0:1] < s)

        X_above_batch = X_batch_original[mask_above[:, 0]]
        X_below_batch = X_batch_original[mask_below[:, 0]]

        # Normalizing minibatches back
        X_above_batch = (X_above_batch - self.mu_X) / self.sigma_X
        X_below_batch = (X_below_batch - self.mu_X) / self.sigma_X

        return X_above_batch, X_below_batch

    def train(self, nIter=10000, batch_size=128):
        start_time = timeit.default_timer()

        for it in range(nIter):
            # Fetch boundary and data mini-batches
            X_0_batch, s_0_batch = self.fetch_minibatch(self.bcs_sampler, batch_size)
            X_bc_batch, u_bc_batch = self.fetch_minibatch(self.Sbc_sampler, batch_size)
            X_SNc_batch, _ = self.fetch_minibatch(self.SNc_sampler, batch_size)
            X_data_batch, u_data_batch = self.fetch_minibatch(self.data_sampler, batch_size)

            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x_u_tf: X_res_batch[:, 0:1], self.t_u_tf: X_res_batch[:, 1:2],
                       self.t_0_tf: X_0_batch[:, 1:2], self.s_0_tf: s_0_batch,
                       self.t_bc_tf: X_bc_batch[:, 1:2], self.s_bc_tf: u_bc_batch,
                       self.t_Nc_tf: X_SNc_batch[:, 1:2],
                       self.x_r_tf: X_res_batch[:, 0:1], self.t_r_tf: X_res_batch[:, 1:2],
                       self.x_data_tf: X_data_batch[:, 0:1], self.t_data_tf: X_data_batch[:, 1:2],
                       self.u_data_tf: u_data_batch}

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

    # Predictions for u
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

    # Predictions for s
    def predict_s(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_u_tf: X_star[:, 1:2]}
        s_star = self.sess.run(self.s_pred, tf_dict)
        return s_star

class Stefan1D_2P_inverse_III:
    def __init__(self, layers_u, layers_s, bcs_sampler, Sbc_sampler, SNc_sampler, res_sampler, data_sampler, method):
        # Normalization constants
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]
        self.mu_t, self.sigma_t = self.mu_X[1], self.sigma_X[1]

        # Samplers
        self.bcs_sampler = bcs_sampler
        self.Sbc_sampler = Sbc_sampler
        self.SNc_sampler = SNc_sampler
        self.res_sampler = res_sampler
        self.data_sampler = data_sampler
        
        # Methpd
        self.method = method

        # Initialize network weights and biases
        self.layers_u = layers_u
        self.weights_u, self.biases_u = self.initialize_NN(layers_u)

        self.layers_s = layers_s
        self.weights_s, self.biases_s = self.initialize_NN(layers_s)
        
        # Adaptive constant
        self.beta = 0.9
        self.adaptive_constant_val = np.array(1.0)

        # Unknown parameters
        self.alpha_1 = tf.Variable(tf.ones([1], dtype=tf.float32) * 0.1, dtype=tf.float32)
        self.alpha_2 = tf.Variable(tf.ones([1], dtype=tf.float32) * 0.1, dtype=tf.float32)

        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Define placeholders and computational graph
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))  # [0, 1]
        self.t_u_tf = tf.placeholder(tf.float32, shape=(None, 1))  # [0, 1]

        self.x_data_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_data_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_data_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_0_tf = tf.placeholder(tf.float32, shape=(None, 1))  # t = 0
        self.s_0_tf = tf.placeholder(tf.float32, shape=(None, 1))  # s(0)

        self.t_bc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.s_bc_tf = tf.placeholder(tf.float32, shape=(None, 1))  # s(0)

        self.x_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_r_below_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_r_below_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_r_above_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_r_above_tf = tf.placeholder(tf.float32, shape=(None, 1))
        
        self.adaptive_constant_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_val.shape)

        # Evaluate predictions
        self.s_pred = self.net_s(self.t_u_tf)
        self.u1_pred, self.u2_pred = self.net_u1u2(self.x_r_tf, self.t_r_tf)
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)

        self.s_bc1_pred, self.s_bc2_pred  = self.net_u1u2((self.net_s(self.t_bc_tf) - self.mu_x) / self.sigma_x, self.t_bc_tf)

        self.r_u1_pred, _ = self.net_r_u1u2(self.x_r_below_tf, self.t_r_below_tf)
        _, self.r_u2_pred = self.net_r_u1u2(self.x_r_above_tf, self.t_r_above_tf)

        self.r_Nc_pred = self.net_r_Nc(self.t_Nc_tf)

        self.u_data_pred = self.net_u(self.x_data_tf, self.t_data_tf)

        # Stefan Boundary loss
        self.loss_bc1 = tf.reduce_mean(tf.square(self.s_bc1_pred))  # u1(s(t),t) = u2(s(t), t) = 0
        self.loss_bc2 = tf.reduce_mean(tf.square(self.s_bc2_pred))  # u1(s(t),t) = u2(s(t), t) = 0
        self.loss_s_0 = tf.reduce_mean(tf.square(self.net_s(self.t_0_tf) - self.s_0_tf))  # s(0) = 0.5
        self.loss_Nc = tf.reduce_mean(tf.square(self.r_Nc_pred))  # Neumann Condition

        # Data loss
        self.loss_data = self.adaptive_constant_tf * tf.reduce_mean(tf.square(self.u_data_pred - self.u_data_tf))

        # Boundary loss
        self.loss_bcs = self.loss_bc1 + self.loss_bc2 + self.loss_s_0 + self.loss_Nc

        # Residual loss
        self.loss_res_u1 = tf.reduce_mean(tf.square(self.r_u1_pred))
        self.loss_res_u2 = tf.reduce_mean(tf.square(self.r_u2_pred))
        self.loss_res = self.loss_res_u1 + self.loss_res_u2

        # Total loss
        self.loss = self.loss_res + self.loss_data + self.loss_bcs

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
        
        self.alpha_1_log = []
        self.alpha_2_log = []
        
        self.saver = tf.train.Saver()

        # Generate dicts for gradients storage
        self.dict_gradients_res_layers = self.generate_grad_dict(self.layers_u)
        self.dict_gradients_data_layers = self.generate_grad_dict(self.layers_u)

        # Gradients Storage
        self.grad_res = []
        self.grad_data = []
        for i in range(len(self.layers_u) - 1):
            self.grad_res.append(tf.gradients(self.loss_res, self.weights_u[i],  unconnected_gradients='zero')[0])
            self.grad_data.append(tf.gradients(self.loss_data, self.weights_u[i], unconnected_gradients='zero')[0])
            
        # Compute and store the adaptive constant
        self.adpative_constant_log = []
        self.adaptive_constant_list = []

        self.max_grad_res_list = []
        self.mean_grad_data_list = []

        for i in range(len(self.layers_u) - 1):
            self.max_grad_res_list.append(tf.reduce_max(tf.abs(self.grad_res[i])))
            self.mean_grad_data_list.append(tf.reduce_mean(tf.abs(self.grad_data[i])))

        self.max_grad_res = tf.reduce_max(tf.stack(self.max_grad_res_list))
        self.mean_grad_data = tf.reduce_mean(tf.stack(self.mean_grad_data_list))
        self.adaptive_constant = self.max_grad_res / self.mean_grad_data

        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    # Create dictionary to store gradients
    def generate_grad_dict(self, layers):
        num = len(layers) - 1
        grad_dict = {}
        for i in range(num):
            grad_dict['layer_{}'.format(i + 1)] = []
        return grad_dict

    # Save gradients
    def save_gradients(self, tf_dict):
        num_layers = len(self.layers_u)
        for i in range(num_layers - 1):
            grad_res_value, grad_data_value = self.sess.run(
                [self.grad_res[i], self.grad_data[i]], feed_dict=tf_dict)

            # save gradients of loss_res and loss_bcs
            self.dict_gradients_res_layers['layer_' + str(i + 1)].append(grad_res_value.flatten())
            self.dict_gradients_data_layers['layer_' + str(i + 1)].append(grad_data_value.flatten())
        return None

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
    def forward_pass(self, H, weights, biases):
        num_layers = len(weights)
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

    # Forward pass for u1, u2
    def net_u1u2(self, x, t):
        u1u2 = self.forward_pass(tf.concat([x, t], 1), self.weights_u, self.biases_u)
        u1 = u1u2[:,0:1]
        u2 = u1u2[:,1:2]
        return u1, u2

    def net_u1u2_x(self, x, t):
        u1, u2 = self.net_u1u2(x, t)
        u1_x = tf.gradients(u1, x)[0] / self.sigma_x
        u2_x = tf.gradients(u2, x)[0] / self.sigma_x
        return u1_x, u2_x

    # Forward pass for s
    def net_s(self, t):
        s = self.forward_pass(t, self.weights_s, self.biases_s)
        return s

    # Forward pass for u
    def net_u(self, x, t):
        u1,u2 = self.net_u1u2(x, t)
        s = self.net_s(t)  # consider normalization

        # demoralizing x coordinates
        x_hat = x * self.sigma_x + self.mu_x
        # consider s = x_hat
        u = tf.multiply(u1, 0.5 * (tf.sign(s - x_hat) + 1)) + tf.multiply(u2, 0.5 * (tf.sign(x_hat - s) + 1))
        return u

    # Forward pass for residual
    def net_r_u1u2(self, x, t):
        u1, u2 = self.net_u1u2(x, t)
        u1_t = tf.gradients(u1, t)[0] / self.sigma_t
        u2_t = tf.gradients(u2, t)[0] / self.sigma_t
        
        u1_x = tf.gradients(u1, x)[0] / self.sigma_x
        u2_x = tf.gradients(u2, x)[0] / self.sigma_x
        
        u1_xx = tf.gradients(u1_x, x)[0] / self.sigma_x
        u2_xx = tf.gradients(u2_x, x)[0] / self.sigma_x
        
        r_u1 = u1_t - self.alpha_1 * u1_xx
        r_u2 = u2_t - self.alpha_2 * u2_xx
        return r_u1, r_u2


    def net_r_Nc(self, t):
        s = self.net_s(t)
        s_t = tf.gradients(s, t)[0] / self.sigma_t

        # Normalizing s
        s = (s - self.mu_x) / self.sigma_x

        u1_x, u2_x = self.net_u1u2_x(s, t)

        residual = s_t - u2_x + 2 * u1_x
        return residual

    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def split_minibatch(self, X_batch):
        
        # denormalizing minibatches
        X_batch_original = X_batch * self.sigma_X + self.mu_X
        
        s = self.predict_s(X_batch)
        
        mask_above = (X_batch_original[:, 0:1] >= s)
        mask_below = (X_batch_original[:, 0:1] < s)

        X_above_batch = X_batch_original[mask_above[:, 0]]
        X_below_batch = X_batch_original[mask_below[:, 0]]

        # Normalizing minibatches back
        X_above_batch = (X_above_batch - self.mu_X) / self.sigma_X
        X_below_batch = (X_below_batch - self.mu_X) / self.sigma_X

        return X_below_batch, X_above_batch

    def train(self, nIter=10000, batch_size=128):
        start_time = timeit.default_timer()

        for it in range(nIter):
            # Fetch boundary and data mini-batches
            X_0_batch, s_0_batch = self.fetch_minibatch(self.bcs_sampler, batch_size)
            X_bc_batch, u_bc_batch = self.fetch_minibatch(self.Sbc_sampler, batch_size)
            X_SNc_batch, _ = self.fetch_minibatch(self.SNc_sampler, batch_size)
            X_data_batch, u_data_batch = self.fetch_minibatch(self.data_sampler, batch_size)

            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)
            
            X_below_batch, X_above_batch = self.split_minibatch(X_res_batch)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x_u_tf: X_res_batch[:, 0:1], self.t_u_tf: X_res_batch[:, 1:2],
                       self.t_0_tf: X_0_batch[:, 1:2], self.s_0_tf: s_0_batch,
                       self.t_bc_tf: X_bc_batch[:, 1:2], self.s_bc_tf: u_bc_batch,
                       self.t_Nc_tf: X_SNc_batch[:, 1:2],
                       self.x_r_tf: X_res_batch[:, 0:1], self.t_r_tf: X_res_batch[:, 1:2],
                       self.x_r_below_tf: X_below_batch[:,0:1], self.t_r_below_tf: X_below_batch[:,1:2], 
                       self.x_r_above_tf: X_above_batch[:,0:1], self.t_r_above_tf: X_above_batch[:,1:2], 
                       self.x_data_tf: X_data_batch[:, 0:1], self.t_data_tf: X_data_batch[:, 1:2],
                       self.u_data_tf: u_data_batch,
                       self.adaptive_constant_tf: self.adaptive_constant_val}

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

                alpha_1_value, alpha_2_value = self.sess.run([self.alpha_1, self.alpha_2])
                print('It: %d, Loss: %.3e, Loss_bcs: %.3e, Loss_Data: %.3e, Loss_res: %.3e, Time: %.2f' %
                      (it, loss_value, loss_bcs_value, loss_data_value, loss_res_value, elapsed))
                
                print('Alpha_1: {}, Alpha_2: {}'.format(alpha_1_value, alpha_2_value))
                start_time = timeit.default_timer()
                print('Adaptive Constant: {:.3f}'.format(self.adaptive_constant_val))
                
                self.alpha_1_log.append(alpha_1_value)
                self.alpha_2_log.append(alpha_2_value)
                
                if self.method in ['M2']:
                    adaptive_constant_value = self.sess.run(self.adaptive_constant, tf_dict)
                    self.adaptive_constant_val = adaptive_constant_value * (1.0 - self.beta) \
                                                 + self.beta * self.adaptive_constant_val
                self.adpative_constant_log.append(self.adaptive_constant_val)

    # Predictions for u
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

    # Predictions for s
    def predict_s(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_u_tf: X_star[:, 1:2]}
        s_star = self.sess.run(self.s_pred, tf_dict)
        return s_star






