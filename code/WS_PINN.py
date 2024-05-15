"""
Created by 
@author Shengze Cai

Modified by
@author Mitchell Daneker

Contact Mitchell Daneker with any questions via mdaneker@seas.upenn.edu
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

if tf.__version__ >= "2.0.0":
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
import numpy as np
import scipy.io
import time
import math
from NSFnets3D import *


# ====================================================================
# spatial resolution: how many slices are used, different for each aneurysm

# old notation called each aneurysm by its mouth size, in case we missed
# renaming anywhere and you see it that is why, placing here for claification.
# Aneurysm #1 was medium, Aneurysm #2 was large, and Aneurysm #3 was small

aneurysm_to_use = 1

if aneurysm_to_use == 1:
    # For aneurysm #1:
    num_slice = 39  # can be selected from [7,15,23,31,39]
    dimensionless_time = 11.3  # for one period
    Rey = 1 / 0.0307
    num_interval_total = 29  # fixed 29 snapshots in total - [0,28]
elif aneurysm_to_use == 2:
    # For aneurysm #2:
    num_slice = 59  # can be selected from [11,23,35,47,59]
    dimensionless_time = 10.0  # for one period
    Rey = 1 / 0.041066
    num_interval_total = 31  # fixed 31 snapshots in total - [0,30]
elif aneurysm_to_use == 3:
    # For aneurysm #3:
    num_slice = 39  # can be selected from [9,18,27,36,45]
    dimensionless_time = 11.0  # for one period
    Rey = 1 / 0.0411437
    num_interval_total = 31  # fixed 31 snapshots in total - [0,30]
else:
    print(
        "Aneursym_to_use not found, you will encounter an error loading data"
    )

# temporal resolution, must be > 2 to include first and last snapshot
num_snapshot = 29  # can be any number between 2 and num_interval_total


# Where your data is located, if in the same directery leave as ""
data_fileDir = ""

# ====================================================================
# NN hyper-parameters
# define the network architecture
num_layer = 8
num_node = 150
# 4 inputs (t,x,y,z), 4 outputs (u,v,w,p)
layers = [4] + num_layer * [num_node] + [4]

# ====================================================================
# ====================  saving settings  =============================
# ====================================================================

Model = "PINN_MA3D"
data_interval = "NOslice_" + str(num_slice) + "_NOtime_" + str(num_snapshot)
folderName = Model + "/" + data_interval

current_directory = os.getcwd()

relative_path = "/PINNresults/" + folderName + "/"
save_results_to = current_directory + relative_path
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

relative_path = "/PINNmodels/" + folderName + "/"
save_models_to = current_directory + relative_path
if not os.path.exists(save_models_to):
    os.makedirs(save_models_to)

# ====================================================================
# ========================  Load data  ===============================
# ====================================================================
print("\n\nLoading data ...\n\n")


# ====================================================================
# load observable data
data_fileName = data_fileDir + "observables_" + str(num_slice) + ".npz"
data = np.load(data_fileName)["arr_0"]
# downsampling in time
times = np.arange(
    0,
    int(num_interval_total - 1),
    math.ceil(int(num_interval_total - 1) / (num_snapshot - 1)),
)
times = np.append(times, num_interval_total)
while len(times) < num_snapshot:
    randtime = np.random.randint(1, int(num_interval_total - 2))
    if randtime not in times:
        times = np.append(times, randtime)
times = np.sort(times)
locs = np.where(data[:, 3:4] == times)[0]

X_star = data[locs, 0:1].astype(np.float32).flatten()[:, None]
Y_star = data[locs, 1:2].astype(np.float32).flatten()[:, None]
Z_star = data[locs, 2:3].astype(np.float32).flatten()[:, None]
T_star = (
    data[locs, 3:4].astype(np.float32).flatten()[:, None]
    * dimensionless_time
    / num_interval_total
)
U_star = data[locs, 4:5].astype(np.float32).flatten()[:, None]
V_star = data[locs, 5:6].astype(np.float32).flatten()[:, None]
W_star = data[locs, 6:7].astype(np.float32).flatten()[:, None]

# ====================================================================
# load testing data
fData_fileName = data_fileDir + "newtonian.npz"
fData = np.load(fData_fileName)["arr_0"]
x_f = fData[:, 0:1].astype(np.float32).flatten()[:, None]
y_f = fData[:, 1:2].astype(np.float32).flatten()[:, None]
z_f = fData[:, 2:3].astype(np.float32).flatten()[:, None]
t_f = (
    fData[:, 3:4].astype(np.float32).flatten()[:, None]
    * dimensionless_time
    / num_interval_total
)
u_f = fData[:, 4:5].astype(np.float32).flatten()[:, None]
v_f = fData[:, 5:6].astype(np.float32).flatten()[:, None]
w_f = fData[:, 6:7].astype(np.float32).flatten()[:, None]
p_f = fData[:, 7:8].astype(np.float32).flatten()[:, None]

# ====================================================================
# load boundary points
bData_fileName = data_fileDir + "aneurysm_highres_wallpoints_only.npz"
bData_fileName = data_fileDir + "aneurysm_highres_wallpoints_only.npz"
bcsData = np.load(bData_fileName)["arr_0"]
x_b = bcsData[:, 0:1].astype(np.float32).flatten()[:, None]
y_b = bcsData[:, 1:2].astype(np.float32).flatten()[:, None]
z_b = bcsData[:, 2:3].astype(np.float32).flatten()[:, None]
t_b = (
    np.array(range(0, num_interval_total, 1))
    .astype(np.float32)
    .reshape([1, -1])
)
x_b = np.tile(x_b, [1, t_b.shape[1]])
y_b = np.tile(y_b, [1, t_b.shape[1]])
z_b = np.tile(z_b, [1, t_b.shape[1]])
t_b = np.tile(t_b, [x_b.shape[0], 1])
x_b = x_b.flatten()[:, None]
y_b = y_b.flatten()[:, None]
z_b = z_b.flatten()[:, None]
t_b = t_b.flatten()[:, None] * dimensionless_time / num_interval_total


del data, fData, bcsData


# ====================================================================
# ====================  main function  ===============================
# ====================================================================


def data_warmup():  # This is data only
    lamD = 500
    lamE = 0
    lamB = 10

    # define some network parameters
    N_residual = 1000000  # number of residual points in the domain
    N_b = 500000  # max number of boundary points
    num_gstep = 20000  # total number of training iterations
    batS = 10000  # batch size for each iteration
    lr = 1e-3  # initial learning rate

    print("Size of MRI slices:", T_star.shape)

    # ====================================================================
    # downsample the residual points - no need to use all
    idx = np.random.choice(t_f.shape[0], np.uint32(N_residual), replace=False)
    print("Size of residual:", idx.shape)
    t_train_f = t_f[idx, :]
    x_train_f = x_f[idx, :]
    y_train_f = y_f[idx, :]
    z_train_f = z_f[idx, :]

    # ====================================================================
    # downsample the bc points - no need to use all if too many
    N_b = min([N_b, t_b.shape[0]])
    idx = np.random.choice(t_b.shape[0], np.uint32(N_b), replace=False)
    print("Size of BC:", idx.shape)
    t_train_b = t_b[idx, :]
    x_train_b = x_b[idx, :]
    y_train_b = y_b[idx, :]
    z_train_b = z_b[idx, :]

    # ====================================================================
    # ========================  training  ================================
    # ====================================================================
    model = NSFnets_TXYZ_BCS(
        T_star,
        X_star,
        Y_star,
        Z_star,
        U_star,
        V_star,
        W_star,
        t_train_f,
        x_train_f,
        y_train_f,
        z_train_f,
        t_train_b,
        x_train_b,
        y_train_b,
        z_train_b,
        layers,
        lamD,
        lamE,
        lamB,
        Rey,
    )
    print("\n-----------------------------------")
    print("Reynolds   : %.1f" % (Rey))
    print("N_data     : %d" % (T_star.shape[0]))
    print("N_residual : %d" % (N_residual))
    print("N_residual : %d" % (N_b))
    print("lamD       : %d" % (model.lambda_data))
    print("lamE       : %d" % (model.lambda_equ))
    print("lamB       : %d" % (model.lambda_bcs))
    print("result saved to :%s" % (save_results_to))
    print("-----------------------------------\n")

    # model.saver.restore(model.sess, save_models_to+"model_uv_1.ckpt")
    model.evaluate_self()

    model.train(num_gstep=num_gstep, batch_size=batS, learning_rate=lr)
    model.evaluate_self()
    loss_log = model.loss_log
    nu_v_log = model.nu_v_log
    model.saver.save(model.sess, save_models_to + "model_uv.ckpt")

    print("\n")
    print("Data warm-up done ......\n")
    # # ====================================================================
    # # =======================  evaluation  ===============================
    # # ====================================================================
    u_data_pred, v_data_pred, w_data_pred, p_data_pred = predict3D(
        model, T_star, X_star, Y_star, Z_star
    )
    scipy.io.savemat(
        save_results_to
        + "prediction_training_%s.mat" % (time.strftime("%d%m%Y_%H%M%S")),
        {
            "U_data_pred": u_data_pred,
            "V_data_pred": v_data_pred,
            "W_data_pred": w_data_pred,
            "P_data_pred": p_data_pred,
        },
    )
    del u_data_pred, v_data_pred, w_data_pred, p_data_pred

    # Final Prediction results
    u_pred, v_pred, w_pred, p_pred = predict3D(model, t_f, x_f, y_f, z_f)
    scipy.io.savemat(
        save_results_to
        + "prediction_testing_%s.mat" % (time.strftime("%d%m%Y_%H%M%S")),
        {
            "U_pred": u_pred,
            "V_pred": v_pred,
            "W_pred": w_pred,
            "P_pred": p_pred,
            "nu_v": nu_v_log,
            "loss": loss_log,
            "num_layer": num_layer,
            "num_node": num_node,
            "batchSize": batS,
        },
    )
    print("\n")
    print("Results saved ......\n")

    error_u = np.linalg.norm(u_f - u_pred, 2) / np.linalg.norm(u_f, 2)
    error_v = np.linalg.norm(v_f - v_pred, 2) / np.linalg.norm(v_f, 2)
    error_w = np.linalg.norm(w_f - w_pred, 2) / np.linalg.norm(w_f, 2)
    Vpred = (u_pred**2 + v_pred**2 + w_pred**2) ** 0.5
    Vtrue = (u_f**2 + v_f**2 + w_f**2) ** 0.5
    error_mag = np.linalg.norm(Vtrue - Vpred, 2) / np.linalg.norm(Vtrue, 2)
    del u_pred, v_pred, w_pred, p_pred, model

    return error_u, error_v, error_w, error_mag


# ====================================================================


def PINN():  # Now we add the physics
    lamD = 500
    lamE = 1
    lamB = 10

    # define some network parameters
    N_residual = 1000000  # number of residual points in the domain
    N_b = 500000  # max number of boundary points
    num_gstep = 10000  # total number of training iterations
    batS = 10000  # batch size for each iteration
    lr = 1e-4  # initial learning rate

    # ====================================================================
    # downsample the residual points - no need to use all
    idx = np.random.choice(t_f.shape[0], np.uint32(N_residual), replace=False)
    t_train_f = t_f[idx, :]
    x_train_f = x_f[idx, :]
    y_train_f = y_f[idx, :]
    z_train_f = z_f[idx, :]

    # ====================================================================
    # downsample the bc points - no need to use all if too many
    N_b = min([N_b, t_b.shape[0]])
    idx = np.random.choice(t_b.shape[0], np.uint32(N_b), replace=False)
    t_train_b = t_b[idx, :]
    x_train_b = x_b[idx, :]
    y_train_b = y_b[idx, :]
    z_train_b = z_b[idx, :]

    # ====================================================================
    # ========================  training  ================================
    # ====================================================================
    model = NSFnets_TXYZ_BCS(
        T_star,
        X_star,
        Y_star,
        Z_star,
        U_star,
        V_star,
        W_star,
        t_train_f,
        x_train_f,
        y_train_f,
        z_train_f,
        t_train_b,
        x_train_b,
        y_train_b,
        z_train_b,
        layers,
        lamD,
        lamE,
        lamB,
        Rey,
    )
    print("\n-----------------------------------")
    print("Reynolds   : %.1f" % (Rey))
    print("N_data     : %d" % (T_star.shape[0]))
    print("N_residual : %d" % (N_residual))
    print("N_residual : %d" % (N_b))
    print("lamD       : %d" % (model.lambda_data))
    print("lamE       : %d" % (model.lambda_equ))
    print("lamB       : %d" % (model.lambda_bcs))
    print("result saved to :%s" % (save_results_to))
    print("-----------------------------------\n")

    model.saver.restore(model.sess, save_models_to + "model_uv.ckpt")

    model.train(num_gstep=num_gstep, batch_size=batS, learning_rate=lr)
    model.evaluate_self()
    loss_log = model.loss_log
    nu_v_log = model.nu_v_log
    model.saver.save(model.sess, save_models_to + "model_uv.ckpt")

    u_pred, v_pred, w_pred, p_pred = predict3D(model, t_f, x_f, y_f, z_f)

    error_u = np.linalg.norm(u_f - u_pred, 2) / np.linalg.norm(u_f, 2)
    error_v = np.linalg.norm(v_f - v_pred, 2) / np.linalg.norm(v_f, 2)
    error_w = np.linalg.norm(w_f - w_pred, 2) / np.linalg.norm(w_f, 2)
    Vpred = (u_pred**2 + v_pred**2 + w_pred**2) ** 0.5
    Vtrue = (u_f**2 + v_f**2 + w_f**2) ** 0.5
    error_mag = np.linalg.norm(Vtrue - Vpred, 2) / np.linalg.norm(Vtrue, 2)

    del u_pred, v_pred, w_pred, p_pred, model

    return error_u, error_v, error_w, error_mag


if __name__ == "__main__":

    graph1 = tf.Graph()
    with graph1.as_default():
        error_u_1, error_v_1, error_w_1, error_mag_1 = data_warmup()

    graph2 = tf.Graph()
    with graph2.as_default():
        error_u_2, error_v_2, error_w_2, error_mag_2 = PINN()

    print("\n")

    print(" Data warm-up: data + bcs: ")
    print("   relative l2 error u:    %e" % (error_u_1))
    print("   relative l2 error v:    %e" % (error_v_1))
    print("   relative l2 error w:    %e" % (error_w_1))
    print("   relative l2 error |V|:  %e" % (error_mag_1))
    print("---------------------------------------\n")

    print(" PINN: data + bcs + equ used: ")
    print("   relative l2 error u:    %e" % (error_u_2))
    print("   relative l2 error v:    %e" % (error_v_2))
    print("   relative l2 error w:    %e" % (error_w_2))
    print("   relative l2 error |V|:  %e" % (error_mag_2))
    print("---------------------------------------\n")
