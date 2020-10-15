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

class Stefan1D_direct:
    def __init__(self, layers_u, layers_s, ics_sampler, Ncs_sampler, res_sampler):
        # Normalization constants
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]
        self.mu_t, self.sigma_t = self.mu_X[1], self.sigma_X[1]

        # Samplers
        self.ics_sampler = ics_sampler
        self.Ncs_sampler = Ncs_sampler
        self.res_sampler = res_sampler

        # Initialize network weights and biases
        self.layers_u = layers_u
        self.weights_u, self.biases_u = self.initialize_NN(layers_u)
        
        self.layers_s = layers_s
        self.weights_s, self.biases_s = self.initialize_NN(layers_s)

        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Define placeholders and computational graph
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.s_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_0_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_0_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_0_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.s_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        # Evaluate predictions
        self.s_pred = self.net_s(self.t_u_tf)
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)

        self.u_0_pred = self.net_u(self.x_0_tf, self.t_0_tf)
        self.u_Sbc_pred = self.net_u((self.s_pred - self.mu_x) / self.sigma_x, self.t_u_tf)
        self.s_0_pred = self.net_s(self.t_0_tf)
        self.u_Nc_pred = self.net_u_x(self.x_Nc_tf, self.t_Nc_tf)

        self.r_u_pred = self.net_r_u(self.x_r_tf, self.t_r_tf)
        self.r_Nc_pred = self.net_r_Nc(self.t_Nc_tf)

        # Boundary loss and Neumann loss
        self.loss_u_0 = tf.reduce_mean(tf.square(self.u_0_pred - self.u_0_tf))
        self.loss_Sbc = tf.reduce_mean(tf.square(self.u_Sbc_pred))
        self.loss_s_0 = tf.reduce_mean(tf.square(self.s_0_pred - (2.0 - np.sqrt(3))))
        self.loss_uNc = tf.reduce_mean(tf.square(self.u_Nc_pred - 2.0))

        # Residual loss
        self.loss_res = tf.reduce_mean(tf.square(self.r_u_pred))
        self.loss_SNc = tf.reduce_mean(tf.square(self.r_Nc_pred - self.s_Nc_tf))

        # Total loss
        self.loss_ics = self.loss_s_0 + self.loss_u_0
        self.loss_bcs = self.loss_Sbc + self.loss_SNc + self.loss_uNc
        self.loss = self.loss_bcs + self.loss_ics + self.loss_res

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        1000, 0.9, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.loss_bcs_log = []
        self.loss_ics_log = []
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

    def net_u(self, x, t):
        u = self.forward_pass_u(tf.concat([x, t], 1))
        return u

    def net_s(self, t):
        s = self.forward_pass_s(t)
        return s

    def net_u_x(self, x, t):
        u = self.net_u(x, t)
        u_x = tf.gradients(u, x)[0] / self.sigma_x
        return u_x

    # Forward pass for residual
    def net_r_u(self, x, t):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0] / self.sigma_t
        u_x = tf.gradients(u, x)[0] / self.sigma_x
        u_xx = tf.gradients(u_x, x)[0] / self.sigma_x
        residual = u_t - u_xx
        return residual

    def net_r_Nc(self, t):
        s = self.net_s(t)

        # Normalize s
        s = (s - self.mu_x) / self.sigma_x

        u_x = self.net_u_x(s, t)
        residual = u_x
        return residual

    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def train(self, nIter=10000, batch_size=128):
        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch boundary and Neumann mini-batches
            X_ics_batch, u_ics_batch = self.fetch_minibatch(self.ics_sampler, batch_size)
            X_Ncs_batch, u_Ncs_batch = self.fetch_minibatch(self.Ncs_sampler, batch_size)
   
            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x_u_tf: X_res_batch[:, 0:1], self.t_u_tf: X_res_batch[:, 1:2],
                       self.x_0_tf: X_ics_batch[:, 0:1], self.t_0_tf: X_ics_batch[:, 1:2],
                       self.u_0_tf: u_ics_batch,
                       self.x_Nc_tf: X_Ncs_batch[:, 0:1], self.t_Nc_tf: X_Ncs_batch[:, 1:2],
                       self.s_Nc_tf: u_Ncs_batch,
                       self.x_r_tf: X_res_batch[:, 0:1], self.t_r_tf: X_res_batch[:, 1:2]}

            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_ics_value, loss_bcs_value, loss_res_value = self.sess.run([self.loss_ics, self.loss_bcs, self.loss_res], tf_dict)
                self.loss_ics_log.append(loss_ics_value)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)

                print('It: %d, Loss: %.3e, Loss_ics: %.3e, Loss_bcs: %.3e, Loss_res: %.3e, Time: %.2f' %
                      (it, loss_value, loss_ics_value, loss_bcs_value, loss_res_value, elapsed))
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
    
    def predict_r_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_u_tf: X_star[:, 1:2]}
        r_u_star = self.sess.run(self.r_u_pred, tf_dict)
        return r_u_star


class Stefan1D_inverse_I:
    def __init__(self, layers_u, ics_sampler, Ncs_sampler, res_sampler):
        # Normalization constants
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]
        self.mu_t, self.sigma_t = self.mu_X[1], self.sigma_X[1]

        # Samplers
        self.ics_sampler = ics_sampler
        self.Ncs_sampler = Ncs_sampler
        self.res_sampler = res_sampler

        # Initialize network weights and biases
        self.layers_u = layers_u
        self.weights_u, self.biases_u = self.initialize_NN(layers_u)

        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Define placeholders and computational graph
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_ic_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_ic_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_ic_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.s_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        # Evaluate predictions
        self.s_pred = self.net_s(self.t_u_tf)  # s is given
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.u_x_pred = self.net_u_x(self.x_u_tf, self.t_u_tf)

        self.u_ic_pred = self.net_u(self.x_ic_tf, self.t_ic_tf)
        self.u_Sbc_pred = self.net_u((self.s_pred - self.mu_x) / self.sigma_x, self.t_u_tf)
        self.u_Nc_pred = self.net_u_x(self.x_Nc_tf, self.t_Nc_tf)

        self.r_u_pred = self.net_r_u(self.x_r_tf, self.t_r_tf)
        self.r_Nc_pred = self.net_r_Nc(self.t_Nc_tf)

        # Boundary loss and Neumann loss
        self.loss_ics = tf.reduce_mean(tf.square(self.u_ic_pred - self.u_ic_tf))
        self.loss_Sbc = tf.reduce_mean(tf.square(self.u_Sbc_pred))

        # Residual loss
        self.loss_res = tf.reduce_mean(tf.square(self.r_u_pred))
        self.loss_SNc = tf.reduce_mean(tf.square(self.r_Nc_pred - self.s_Nc_tf))

        # Total loss
        self.loss_bcs = self.loss_Sbc + self.loss_SNc
        self.loss = self.loss_bcs + self.loss_ics + self.loss_res

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        1000, 0.9, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        # loss loggers
        self.loss_bcs_log = []
        self.loss_ics_log = []
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

    # Forward pass for u
    def net_u(self, x, t):
        u = self.forward_pass_u(tf.concat([x, t], 1))
        return u

    # Forward pass for s
    def net_s(self, t):
        # denormalzie t
        t = t * self.sigma_t + self.mu_t
        s = 2 - tf.math.sqrt(3 - 2 * t)
        return s

    def net_u_x(self, x, t):
        u = self.net_u(x, t)
        u_x = tf.gradients(u, x)[0] / self.sigma_x
        return u_x

    # Forward pass for residual
    def net_r_u(self, x, t):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0] / self.sigma_t
        u_x = tf.gradients(u, x)[0] / self.sigma_x
        u_xx = tf.gradients(u_x, x)[0] / self.sigma_x
        residual = u_t - u_xx
        return residual

    def net_r_Nc(self, t):
        s = self.net_s(t)

        # Normalize s
        s = (s - self.mu_x) / self.sigma_x

        u_x = self.net_u_x(s, t)
        residual = u_x
        return residual

    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def train(self, nIter=10000, batch_size=128):
        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch boundary and Neumann mini-batches
            X_ics_batch, u_ics_batch = self.fetch_minibatch(self.ics_sampler, batch_size)
            X_Ncs_batch, u_Ncs_batch = self.fetch_minibatch(self.Ncs_sampler, batch_size)

            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x_u_tf: X_res_batch[:, 0:1], self.t_u_tf: X_res_batch[:, 1:2],
                       self.x_ic_tf: X_ics_batch[:, 0:1], self.t_ic_tf: X_ics_batch[:, 1:2],
                       self.u_ic_tf: u_ics_batch,
                       self.x_Nc_tf: X_Ncs_batch[:, 0:1], self.t_Nc_tf: X_Ncs_batch[:, 1:2],
                       self.s_Nc_tf: u_Ncs_batch,
                       self.x_r_tf: X_res_batch[:, 0:1], self.t_r_tf: X_res_batch[:, 1:2]}

            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_ics_value, loss_bcs_value, loss_res_value = self.sess.run(
                    [self.loss_ics, self.loss_bcs, self.loss_res], tf_dict)

                self.loss_ics_log.append(loss_ics_value)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)

                print('It: %d, Loss: %.3e, Loss_ics: %.3e, Loss_bcs: %.3e, Loss_res: %.3e, Time: %.2f' %
                      (it, loss_value, loss_ics_value, loss_bcs_value, loss_res_value, elapsed))
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

    # Predictions for u_x
    def predict_u_x(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]}
        r_u_star = self.sess.run(self.u_x_pred, tf_dict)
        return r_u_star


class Stefan1D_inverse_II:
    def __init__(self, layers_u, layers_s, bcs_sampler, Ncs_sampler, res_sampler, data_sampler):
        # Normalization constants
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]
        self.mu_t, self.sigma_t = self.mu_X[1], self.sigma_X[1]

        # Samplers
        self.bcs_sampler = bcs_sampler
        self.Ncs_sampler = Ncs_sampler
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
        self.u_tf = tf.placeholder(tf.float32, shape=(None, 1))  # u(x,t)
        self.s_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_data_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_data_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_data_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_0_tf = tf.placeholder(tf.float32, shape=(None, 1))  # t = 0
        self.s_0_tf = tf.placeholder(tf.float32, shape=(None, 1))  # s(0)

        self.t_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_Nc_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        # Evaluate predictions
        self.s_pred = self.net_s(self.t_u_tf)
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)

        self.u_data_pred = self.net_u(self.x_data_tf, self.t_data_tf)

        self.u_bc_pred = self.net_u((self.s_pred - self.mu_x)/self.sigma_x, self.t_u_tf)
        self.u_Nc_pred = self.net_u_x((self.net_s(self.t_Nc_tf) - self.mu_x)/self.sigma_x, self.t_Nc_tf)
        self.r_u_pred = self.net_r_u(self.x_r_tf, self.t_r_tf)

        # Stefan Boundary loss
        self.loss_bc = tf.reduce_mean(tf.square(self.u_bc_pred))
        self.loss_s_0 = tf.reduce_mean(tf.square(self.net_s(self.t_0_tf)- self.s_0_tf) )  # s(0) = 0
        self.loss_Nc = tf.reduce_mean(tf.square(self.u_Nc_pred - self.u_Nc_tf))
        # Data loss
        self.loss_data = tf.reduce_mean(tf.square(self.u_data_pred - self.u_data_tf))

        # Boundary loss
        self.loss_bcs = self.loss_bc + self.loss_s_0 + self.loss_Nc
        # Neumann condition is important!

        # Residual loss
        self.loss_res = tf.reduce_mean(tf.square(self.r_u_pred))

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

    # Evaluates the forward pass u
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

    # Evaluates the forward pass s
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
    def net_u(self, x, t):
        u = self.forward_pass_u(tf.concat([x, t], 1))
        return u

    # Forward pass for s
    def net_s(self, t):
      s = self.forward_pass_s(t)
      return s

    def net_s_t(self, t):
       s = self.net_s(t)
       s_t = tf.gradients(s, t)[0] / self.sigma_t
       return s_t

    def net_u_x(self, x, t):
        u = self.net_u(x, t)
        u_x = tf.gradients(u, x)[0] / self.sigma_x
        return u_x

    # Forward pass for residual
    def net_r_u(self, x, t):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0] / self.sigma_t
        u_x = tf.gradients(u, x)[0] / self.sigma_x
        u_xx = tf.gradients(u_x, x)[0] / self.sigma_x
        residual = u_t - u_xx
        return residual

    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def train(self, nIter=10000, batch_size=128):
        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch boundary and data mini-batches
            X_0_batch, s_0_batch = self.fetch_minibatch(self.bcs_sampler, batch_size)
            X_Nc_batch, u_Nc_batch = self.fetch_minibatch(self.Ncs_sampler, batch_size)
            X_data_batch, u_data_batch = self.fetch_minibatch(self.data_sampler, batch_size)

            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x_u_tf: X_res_batch[:, 0:1], self.t_u_tf: X_res_batch[:, 1:2],
                       self.t_Nc_tf: X_Nc_batch[:, 1:2], self.u_Nc_tf: u_Nc_batch,
                       self.t_0_tf: X_0_batch[:, 1:2], self.s_0_tf: s_0_batch,
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

    # Predictions for u_x
    def predict_u_x(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_Nc_tf: X_star[:, 1:2]}
        r_u_star = self.sess.run(self.u_Nc_pred, tf_dict)
        return r_u_star

