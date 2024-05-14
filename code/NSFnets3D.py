"""
@author: Shengze Cai
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

if tf.__version__ >= "2.0.0":
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
import numpy as np
import time


# ====================================================================
# ===============  evaluations for large data ========================
# ====================================================================
def predict3D(
    model,
    T_star,
    X_star,
    Y_star,
    Z_star,
    U_star=None,
    V_star=None,
    W_star=None,
):
    U_pred = np.zeros_like(X_star)
    V_pred = np.zeros_like(X_star)
    W_pred = np.zeros_like(X_star)
    P_pred = np.zeros_like(X_star)
    error_u = 0.0
    error_v = 0.0
    error_w = 0.0
    N_f = T_star.shape[0]
    batch_size = 100000
    iii = 1
    for it in range(0, N_f, batch_size):
        if it + batch_size < N_f:
            idx = np.arange(it, it + batch_size)
        else:
            idx = np.arange(it, N_f)
        t_test = T_star[idx, :].reshape(-1, 1)
        x_test = X_star[idx, :].reshape(-1, 1)
        y_test = Y_star[idx, :].reshape(-1, 1)
        z_test = Z_star[idx, :].reshape(-1, 1)
        # Prediction
        u_pred, v_pred, w_pred, p_pred = model.predict(
            t_test, x_test, y_test, z_test
        )
        U_pred[idx, :] = u_pred.reshape(-1, 1)
        V_pred[idx, :] = v_pred.reshape(-1, 1)
        W_pred[idx, :] = w_pred.reshape(-1, 1)
        P_pred[idx, :] = p_pred.reshape(-1, 1)

        # Error
        if U_star is not None:
            u_test = U_star[idx, :].reshape(-1, 1)
            v_test = V_star[idx, :].reshape(-1, 1)
            w_test = W_star[idx, :].reshape(-1, 1)
            error_u += np.linalg.norm(u_test - u_pred, 2) / np.linalg.norm(
                u_test, 2
            )
            error_v += np.linalg.norm(v_test - v_pred, 2) / np.linalg.norm(
                v_test, 2
            )
            error_w += np.linalg.norm(w_test - w_pred, 2) / np.linalg.norm(
                w_test, 2
            )
            iii += 1
    if U_star is not None:
        print("\n-----------------------------------")
        print("Mean Error u: %e" % (error_u / iii))
        print("Mean Error v: %e" % (error_v / iii))
        print("Mean Error w: %e" % (error_w / iii))
        print("-----------------------------------\n")
    return U_pred, V_pred, W_pred, P_pred


# ====================================================================
# =============================  Model ===============================
# ====================================================================
######################################################################
#######################  NSFnets_TXYZ_BCS   ##########################
######################################################################
class NSFnets_TXYZ_BCS:
    # Initialize the class
    def __init__(
        self,
        t,
        x,
        y,
        z,
        u,
        v,
        w,
        t_f,
        x_f,
        y_f,
        z_f,
        t_b,
        x_b,
        y_b,
        z_b,
        layers,
        lamD,
        lamE,
        lamB,
        Rey=None,
    ):
        # min and max - for normalization
        X = np.concatenate([t_f, x_f, y_f, z_f], 1)
        self.lb = X.min(0)
        self.ub = X.max(0)

        # preparing data
        self.t = t
        self.x = x
        self.y = y
        self.z = z
        self.u = u
        self.v = v
        self.w = w
        self.t_f = t_f
        self.x_f = x_f
        self.y_f = y_f
        self.z_f = z_f
        self.t_b = t_b
        self.x_b = x_b
        self.y_b = y_b
        self.z_b = z_b

        # Initialize parameters - Reynolds
        self.Rey = Rey
        if self.Rey is not None:
            self.nu_v = 1 / self.Rey  # tf.Variable([0.1], dtype=tf.float32)
            self.P_SCALE = 1
        else:
            self.nu_v = tf.Variable([0.1], dtype=tf.float32)
            self.nu_v = 1e-4 + (1e2 - 1e-4) * tf.nn.sigmoid(self.nu_v)
            # self.P_SCALE = 1
            self.P_SCALE = 1e4
        self.nu_v_log = []

        # layers
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True
            )
        )
        self.saver = tf.train.Saver()
        self.t_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.z_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.w_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_b_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_b_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_b_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.z_b_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.dummy_f_tf = tf.placeholder(tf.float32, shape=(None, layers[-1]))
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.z_f_tf = tf.placeholder(tf.float32, shape=[None, 1])

        # physics informed neural networks
        # evaluate the data points
        (
            self.u_pred,
            self.v_pred,
            self.w_pred,
            self.p_pred,
        ) = self.forward_net(self.t_tf, self.x_tf, self.y_tf, self.z_tf)
        # evaluate the boundary points
        (self.u_b_pred, self.v_b_pred, self.w_b_pred, _) = self.forward_net(
            self.t_b_tf, self.x_b_tf, self.y_b_tf, self.z_b_tf
        )
        # evaluate the residual points
        (
            self.eq1_pred,
            self.eq2_pred,
            self.eq3_pred,
            self.eq4_pred,
        ) = self.net_equation(
            self.t_f_tf, self.x_f_tf, self.y_f_tf, self.z_f_tf
        )

        # compute the vorticity
        (
            self.omegaX_pred,
            self.omegaY_pred,
            self.omegaZ_pred,
        ) = self.compute_curl(self.t_tf, self.x_tf, self.y_tf, self.z_tf)

        # loss
        self.loss_log = []
        self.lambda_data = lamD
        self.lambda_equ = lamE
        self.lambda_bcs = lamB
        self.loss_data = (
            tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
            + tf.reduce_mean(tf.square(self.v_tf - self.v_pred))
            + tf.reduce_mean(tf.square(self.w_tf - self.w_pred)) / 5
        )
        self.loss_bcs = (
            tf.reduce_mean(tf.square(self.u_b_pred))
            + tf.reduce_mean(tf.square(self.v_b_pred))
            + tf.reduce_mean(tf.square(self.w_b_pred)) / 5
        )
        self.loss_equ = (
            tf.reduce_mean(tf.square(self.eq1_pred))
            + tf.reduce_mean(tf.square(self.eq2_pred))
            + tf.reduce_mean(tf.square(self.eq3_pred))
            + tf.reduce_mean(tf.square(self.eq4_pred)) * 10
        )

        if lamE == 0:
            self.loss = (
                self.lambda_data * self.loss_data
                + self.lambda_bcs * self.loss_bcs
            )
        else:
            self.loss = (
                self.lambda_data * self.loss_data
                + self.lambda_bcs * self.loss_bcs
                + self.lambda_equ * self.loss_equ
            )

        # optimizers
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.global_step = tf.Variable(0, trainable=False)
        decay_steps = 10000
        decay_rate = 0.5
        self.decay_learning_rate = tf.train.inverse_time_decay(
            self.learning_rate, self.global_step, decay_steps, decay_rate
        )
        self.optimizer = tf.train.AdamOptimizer(self.decay_learning_rate)
        self.train_op = self.optimizer.minimize(
            self.loss, global_step=self.global_step
        )

        init = tf.global_variables_initializer()
        self.sess.run(init)

    # ===================================================
    # =============== Initialization ====================
    # ===================================================
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(
                tf.zeros([1, layers[l + 1]], dtype=tf.float32),
                dtype=tf.float32,
            )
            weights.append(W)
            biases.append(b)
        return weights, biases

    # ===================================================
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(
            tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev),
            dtype=tf.float32,
        )

    # ===================================================
    # =============== Network ====================
    # ===================================================
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        # H = tf.cast(X,tf.float32)
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    # ===================================================
    # ============== Forward computing ==================
    # ===================================================
    def forward_net(self, t, x, y, z):
        uvwp = self.neural_net(
            tf.concat([t, x, y, z], 1), self.weights, self.biases
        )
        u = uvwp[:, 0:1]
        v = uvwp[:, 1:2]
        w = uvwp[:, 2:3]
        p = uvwp[:, 3:4]
        return u, v, w, p * self.P_SCALE

    # ===================================================
    def fwd_gradients(self, U, x):
        g = tf.gradients(U, x, grad_ys=self.dummy_f_tf)[0]
        return tf.gradients(g, self.dummy_f_tf)[0]

    # ===================================================
    def net_equation(self, t, x, y, z):
        nu_v = self.nu_v
        uvwp = self.neural_net(
            tf.concat([t, x, y, z], 1), self.weights, self.biases
        )
        uvwp_t = self.fwd_gradients(uvwp, t)
        uvwp_x = self.fwd_gradients(uvwp, x)
        uvwp_y = self.fwd_gradients(uvwp, y)
        uvwp_z = self.fwd_gradients(uvwp, z)
        uvwp_xx = self.fwd_gradients(uvwp_x, x)
        uvwp_yy = self.fwd_gradients(uvwp_y, y)
        uvwp_zz = self.fwd_gradients(uvwp_z, z)
        u = uvwp[:, 0:1]
        v = uvwp[:, 1:2]
        w = uvwp[:, 2:3]
        p = uvwp[:, 3:4]
        u_t = uvwp_t[:, 0:1]
        v_t = uvwp_t[:, 1:2]
        w_t = uvwp_t[:, 2:3]
        u_x = uvwp_x[:, 0:1]
        v_x = uvwp_x[:, 1:2]
        w_x = uvwp_x[:, 2:3]
        p_x = uvwp_x[:, 3:4]
        u_y = uvwp_y[:, 0:1]
        v_y = uvwp_y[:, 1:2]
        w_y = uvwp_y[:, 2:3]
        p_y = uvwp_y[:, 3:4]
        u_z = uvwp_z[:, 0:1]
        v_z = uvwp_z[:, 1:2]
        w_z = uvwp_z[:, 2:3]
        p_z = uvwp_z[:, 3:4]
        u_xx = uvwp_xx[:, 0:1]
        v_xx = uvwp_xx[:, 1:2]
        w_xx = uvwp_xx[:, 2:3]
        u_yy = uvwp_yy[:, 0:1]
        v_yy = uvwp_yy[:, 1:2]
        w_yy = uvwp_yy[:, 2:3]
        u_zz = uvwp_zz[:, 0:1]
        v_zz = uvwp_zz[:, 1:2]
        w_zz = uvwp_zz[:, 2:3]
        eq1 = (
            u_t
            + (u * u_x + v * u_y + w * u_z)
            + p_x * self.P_SCALE
            - (nu_v) * (u_xx + u_yy + u_zz)
        )
        eq2 = (
            v_t
            + (u * v_x + v * v_y + w * v_z)
            + p_y * self.P_SCALE
            - (nu_v) * (v_xx + v_yy + v_zz)
        )
        eq3 = (
            w_t
            + (u * w_x + v * w_y + w * w_z)
            + p_z * self.P_SCALE
            - (nu_v) * (w_xx + w_yy + w_zz)
        )
        eq4 = u_x + v_y + w_z
        return eq1, eq2, eq3, eq4

    # ===================================================
    def compute_curl(self, t, x, y, z):
        uvwp = self.neural_net(
            tf.concat([t, x, y, z], 1), self.weights, self.biases
        )
        uvwp_x = self.fwd_gradients(uvwp, x)
        uvwp_y = self.fwd_gradients(uvwp, y)
        uvwp_z = self.fwd_gradients(uvwp, z)
        # u_x = uvwp_x[:,0:1]
        v_x = uvwp_x[:, 1:2]
        w_x = uvwp_x[:, 2:3]
        u_y = uvwp_y[:, 0:1]
        # v_y = uvwp_y[:,1:2]
        w_y = uvwp_y[:, 2:3]
        u_z = uvwp_z[:, 0:1]
        v_z = uvwp_z[:, 1:2]
        # w_z = uvwp_z[:,2:3]
        OmegaX = w_y - v_z
        OmegaY = u_z - w_x
        OmegaZ = v_x - u_y
        return OmegaX, OmegaY, OmegaZ

    # ===================================================
    def train(self, num_gstep, batch_size, learning_rate):
        start_time = time.time()
        while self.sess.run(self.global_step) < num_gstep:
            Nd = self.x.shape[0]
            Nf = self.x_f.shape[0]
            Nb = self.x_b.shape[0]
            perm = np.random.permutation(Nd)
            perm_f = np.random.permutation(Nf)

            # loop for all residual points
            for it in range(0, Nf, batch_size):
                if it + batch_size < Nf:
                    idx_f = perm_f[np.arange(it, it + batch_size)]
                else:
                    idx_f = perm_f[np.arange(it, Nf)]
                (t_f_batch, x_f_batch, y_f_batch, z_f_batch) = (
                    self.t_f[idx_f, :],
                    self.x_f[idx_f, :],
                    self.y_f[idx_f, :],
                    self.z_f[idx_f, :],
                )
                # loop for all data points
                for itu in range(0, Nd, batch_size):
                    if itu + batch_size < Nd:
                        idx = perm[np.arange(itu, itu + batch_size)]
                    else:
                        idx = perm[np.arange(itu, Nd)]

                    (
                        t_batch,
                        x_batch,
                        y_batch,
                        z_batch,
                        u_batch,
                        v_batch,
                        w_batch,
                    ) = (
                        self.t[idx, :],
                        self.x[idx, :],
                        self.y[idx, :],
                        self.z[idx, :],
                        self.u[idx, :],
                        self.v[idx, :],
                        self.w[idx, :],
                    )

                    # select boundary points
                    idx_b = np.random.choice(
                        Nb, np.min((batch_size, Nb)), replace=False
                    )
                    (t_b_batch, x_b_batch, y_b_batch, z_b_batch) = (
                        self.t_b[idx_b, :],
                        self.x_b[idx_b, :],
                        self.y_b[idx_b, :],
                        self.z_b[idx_b, :],
                    )

                    tf_dict = {
                        self.t_tf: t_batch,
                        self.x_tf: x_batch,
                        self.y_tf: y_batch,
                        self.z_tf: z_batch,
                        self.u_tf: u_batch,
                        self.v_tf: v_batch,
                        self.w_tf: w_batch,
                        self.t_f_tf: t_f_batch,
                        self.x_f_tf: x_f_batch,
                        self.y_f_tf: y_f_batch,
                        self.z_f_tf: z_f_batch,
                        self.t_b_tf: t_b_batch,
                        self.x_b_tf: x_b_batch,
                        self.y_b_tf: y_b_batch,
                        self.z_b_tf: z_b_batch,
                        self.dummy_f_tf: np.ones(
                            (t_f_batch.shape[0], self.layers[-1])
                        ),
                        self.learning_rate: learning_rate,
                    }
                    self.sess.run(self.train_op, tf_dict)
                    gstep = self.sess.run(self.global_step)
                    if gstep > num_gstep:
                        break

                    # Print
                    if gstep == 1 or gstep % 1000 == 0:
                        elapsed = time.time() - start_time
                        if self.lambda_equ == 0:
                            loss_equ = 0
                            (
                                loss_value,
                                loss_data,
                                loss_bcs,
                                learning_rate_value,
                            ) = self.sess.run(
                                [
                                    self.loss,
                                    self.loss_data,
                                    self.loss_bcs,
                                    self.decay_learning_rate,
                                ],
                                tf_dict,
                            )
                        else:
                            (
                                loss_value,
                                loss_data,
                                loss_equ,
                                loss_bcs,
                                learning_rate_value,
                            ) = self.sess.run(
                                [
                                    self.loss,
                                    self.loss_data,
                                    self.loss_equ,
                                    self.loss_bcs,
                                    self.decay_learning_rate,
                                ],
                                tf_dict,
                            )
                        if self.Rey is not None:
                            nu_v_value = (
                                self.nu_v
                            )  # self.sess.run(self.nu_v_bound)
                        else:
                            nu_v_value = self.sess.run(self.nu_v)
                        print(
                            "--------------------------------------------------------------"
                        )
                        print(
                            "gstep: %d, Time: %.2f, Learning Rate: %.1e"
                            % (gstep, elapsed, learning_rate_value)
                        )
                        print(
                            "gstep: %d, Loss: %.3e, Loss_data: %.3e, Loss_equ: %.3e, Loss_bcs: %.3e"
                            % (
                                gstep,
                                loss_value,
                                loss_data,
                                loss_equ,
                                loss_bcs,
                            )
                        )
                        start_time = time.time()

                        self.loss_log.append(
                            [loss_value, loss_data, loss_equ, loss_bcs]
                        )
                        self.nu_v_log.append([nu_v_value])

                    # Print
                    if gstep % 5000 == 0:
                        print("\n\n")
                        self.evaluate_self()

                if gstep > num_gstep:
                    break

    # ===================================================
    def predict(self, t_star, x_star, y_star, z_star):
        tf_dict = {
            self.t_tf: t_star,
            self.x_tf: x_star,
            self.y_tf: y_star,
            self.z_tf: z_star,
        }
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        w_star = self.sess.run(self.w_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        return u_star, v_star, w_star, p_star

    # ===================================================
    def predict_accel(self, t_star, x_star, y_star, z_star):
        tf_dict = {
            self.t_tf: t_star,
            self.x_tf: x_star,
            self.y_tf: y_star,
            self.z_tf: z_star,
            self.dummy_f_tf: np.ones((t_star.shape[0], self.layers[-1])),
        }
        u_t_star = self.sess.run(self.u_t_pred, tf_dict)
        v_t_star = self.sess.run(self.v_t_pred, tf_dict)
        w_t_star = self.sess.run(self.w_t_pred, tf_dict)
        return u_t_star, v_t_star, w_t_star

    # ===================================================
    def predict_curl(self, t_star, x_star, y_star, z_star):
        tf_dict = {
            self.t_tf: t_star,
            self.x_tf: x_star,
            self.y_tf: y_star,
            self.z_tf: z_star,
            self.dummy_f_tf: np.ones((t_star.shape[0], self.layers[-1])),
        }
        omegaX_star = self.sess.run(self.omegaX_pred, tf_dict)
        omegaY_star = self.sess.run(self.omegaY_pred, tf_dict)
        omegaZ_star = self.sess.run(self.omegaZ_pred, tf_dict)
        return omegaX_star, omegaY_star, omegaZ_star

    # ===================================================
    def predict_residuals(self, t_star, x_star, y_star, z_star):
        tf_dict = {
            self.t_f_tf: t_star,
            self.x_f_tf: x_star,
            self.y_f_tf: y_star,
            self.z_f_tf: z_star,
            self.dummy_f_tf: np.ones((t_star.shape[0], self.layers[-1])),
        }
        eq1_pred = self.sess.run(self.eq1_pred, tf_dict)
        eq2_pred = self.sess.run(self.eq2_pred, tf_dict)
        eq3_pred = self.sess.run(self.eq3_pred, tf_dict)
        eq4_pred = self.sess.run(self.eq4_pred, tf_dict)
        return eq1_pred, eq2_pred, eq3_pred, eq4_pred

    # ===================================================
    def evaluate_self(
        self,
    ):
        """testing all points in the domain"""
        t_test = self.t.reshape(-1, 1)
        x_test = self.x.reshape(-1, 1)
        y_test = self.y.reshape(-1, 1)
        z_test = self.z.reshape(-1, 1)
        u_test = self.u.reshape(-1, 1)
        v_test = self.v.reshape(-1, 1)
        w_test = self.w.reshape(-1, 1)
        # Prediction
        u_pred, v_pred, w_pred, _ = self.predict(
            t_test, x_test, y_test, z_test
        )
        # Error
        error_u = np.linalg.norm(u_test - u_pred, 2) / np.linalg.norm(
            u_test, 2
        )
        error_v = np.linalg.norm(v_test - v_pred, 2) / np.linalg.norm(
            v_test, 2
        )
        error_w = np.linalg.norm(w_test - w_pred, 2) / np.linalg.norm(
            w_test, 2
        )
        print("---------------------------")
        print("Error u: %e" % (error_u))
        print("Error v: %e" % (error_v))
        print("Error w: %e" % (error_w))
        print("---------------------------")

    # ===================================================
    def evaluate(self, T_star, X_star, Y_star, Z_star, U_star, V_star, W_star):
        error_u = 0.0
        error_v = 0.0
        error_w = 0.0
        N_f = T_star.shape[0]
        batch_size = 50000
        iii = 1
        for it in range(0, N_f, batch_size):
            if it + batch_size < N_f:
                idx = np.arange(it, it + batch_size)
            else:
                idx = np.arange(it, N_f)
            t_test = T_star[idx, :].reshape(-1, 1)
            x_test = X_star[idx, :].reshape(-1, 1)
            y_test = Y_star[idx, :].reshape(-1, 1)
            z_test = Z_star[idx, :].reshape(-1, 1)
            u_test = U_star[idx, :].reshape(-1, 1)
            v_test = V_star[idx, :].reshape(-1, 1)
            w_test = W_star[idx, :].reshape(-1, 1)
            # Prediction
            u_pred, v_pred, w_pred, _ = self.predict(
                t_test, x_test, y_test, z_test
            )
            # Error
            error_u += np.linalg.norm(u_test - u_pred, 2) / np.linalg.norm(
                u_test, 2
            )
            error_v += np.linalg.norm(v_test - v_pred, 2) / np.linalg.norm(
                v_test, 2
            )
            error_w += np.linalg.norm(w_test - w_pred, 2) / np.linalg.norm(
                w_test, 2
            )
            iii += 1
        print("\n-----------------------------------")
        print("Errors for training data: ")
        print(" Mean Error u: %e" % (error_u / iii))
        print(" Mean Error v: %e" % (error_v / iii))
        print(" Mean Error w: %e" % (error_w / iii))
        print("-----------------------------------\n")


# %%
