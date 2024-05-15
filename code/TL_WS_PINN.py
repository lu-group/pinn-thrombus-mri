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

stretch_amount = "10"  # "10","20","30","40"

if stretch_amount == "10":
    # For the aneurysm stretched 10% in the z-direction:
    num_slice = 39  # can be selected from [7,15,23,31,39]
    old_num_slice = 39  # can be selected from [7,15,23,31,39]
    old_newtonian_file_name = "newtonian_00"
    newtonian_file_name = "newtonian_10"
if stretch_amount == "20":
    # For the aneurysm stretched 20% in the z-direction:
    num_slice = 59  # can be selected from [11,23,35,47,59]
    old_num_slice = 39  # can be selected from [7,15,23,31,39]
    old_newtonian_file_name = "newtonian_10"
    newtonian_file_name = "newtonian_20"
if stretch_amount == "30":
    # For the aneurysm stretched 30% in the z-direction:
    num_slice = 39  # can be selected from [9,18,27,36,45]
    old_num_slice = 39  # can be selected from [7,15,23,31,39]
    old_newtonian_file_name = "newtonian_20"
    newtonian_file_name = "newtonian_30"
if stretch_amount == "40":
    # For the aneurysm stretched 40% in the z-direction:
    num_slice = 39  # can be selected from [9,18,27,36,45]
    old_num_slice = 39  # can be selected from [7,15,23,31,39]
    old_newtonian_file_name = "newtonian_30"
    newtonian_file_name = "newtonian_40"
else:
    print("stretch_amount not found, you will encounter an error loading data")

# temporal resolution, must be > 2 to include first and last snapshot
num_snapshot = 31  # can be any number between 2 and num_interval_total

dimensionless_time = 11.3  # for one period
Rey = 1 / 0.0307
num_interval_total = 29  # fixed 29 snapshots in total - [0,29]
# Note that the stretched cases actually have more snapshots than the original
# medium sized aneurysm (31 vs. 29), we only use 29 here

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

relative_path = "/PINNmodels/" + folderName + "/tranfer_learning/"
save_transfer_models_to = current_directory + relative_path
if not os.path.exists(save_models_to):
    os.makedirs(save_models_to)

# ====================================================================
# ========================  Load data  ===============================
# ====================================================================
print("\n\nLoading data ...\n\n")


# ====================================================================
# load old observable data
data_fileName = data_fileDir + "observables_" + str(old_num_slice) + ".npz"
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

# load new observable data
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

X_star_stretch = data[locs, 0:1].astype(np.float32).flatten()[:, None]
Y_star_stretch = data[locs, 1:2].astype(np.float32).flatten()[:, None]
Z_star_stretch = data[locs, 2:3].astype(np.float32).flatten()[:, None]
T_star_stretch = (
    data[locs, 3:4].astype(np.float32).flatten()[:, None]
    * dimensionless_time
    / num_interval_total
)
U_star_stretch = data[locs, 4:5].astype(np.float32).flatten()[:, None]
V_star_stretch = data[locs, 5:6].astype(np.float32).flatten()[:, None]
W_star_stretch = data[locs, 6:7].astype(np.float32).flatten()[:, None]

# ====================================================================
# load testing data

# First load the old data for the boundary warmup
fData_fileName = data_fileDir + old_newtonian_file_name
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


# Now load the new data for the stretched domain
fData_fileName = data_fileDir + newtonian_file_name
fData = np.load(fData_fileName)["arr_0"]
x_f_stretch = fData[:, 0:1].astype(np.float32).flatten()[:, None]
y_f_stretch = fData[:, 1:2].astype(np.float32).flatten()[:, None]
z_f_stretch = fData[:, 2:3].astype(np.float32).flatten()[:, None]
t_f_stretch = (
    fData[:, 3:4].astype(np.float32).flatten()[:, None]
    * dimensionless_time
    / num_interval_total
)
u_f_stretch = fData[:, 4:5].astype(np.float32).flatten()[:, None]
v_f_stretch = fData[:, 5:6].astype(np.float32).flatten()[:, None]
w_f_stretch = fData[:, 6:7].astype(np.float32).flatten()[:, None]
p_f_stretch = fData[:, 7:8].astype(np.float32).flatten()[:, None]


# ====================================================================
# load boundary points

# We only need the medium aneurysm points, we can stretch them manually
bData_fileName = data_fileDir + "aneurysm_highres_wallpoints_only.npz"
bcsData = np.load(bData_fileName)["arr_0"]
x_b = bcsData[:, 0:1].astype(np.float32).flatten()[:, None]
y_b = bcsData[:, 1:2].astype(np.float32).flatten()[:, None]
z_b = bcsData[:, 2:3].astype(np.float32).flatten()[:, None]

# We need to adjust the z-axis based on the size of the aneurysm. We used a
# linear scaling so we can just do the following.

z_b = (z_b - z_b.min()) / (z_b.max() - z_b.min()) * (
    z_f.max() - z_f.min()
) + z_f.min()

z_b_stretch = (z_b - z_b.min()) / (z_b.max() - z_b.min()) * (
    z_f_stretch.max() - z_f_stretch.min()
) + z_f_stretch.min()

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

# Only z is stretched in the new case but we apply to x and y to not mess up our
# current values
stretch_diff = (z_f_stretch.max() - z_f_stretch.min()) / (
    z_f.max() - z_f.min()
)

t_b_warm = np.vstack((t_b, t_b, t_b, t_b, t_b))
x_b_warm = np.vstack(
    (
        x_b,
        0.25 * stretch_diff * x_b,
        0.5 * stretch_diff * x_b,
        0.75 * stretch_diff * x_b,
        stretch_diff * x_b,
    )
)
y_b_warm = np.vstack(
    (
        y_b,
        0.25 * stretch_diff * y_b,
        0.5 * stretch_diff * y_b,
        0.75 * stretch_diff * y_b,
        stretch_diff * y_b,
    )
)
z_b_warm = np.vstack(
    (
        z_b,
        0.25 * stretch_diff * z_b,
        0.5 * stretch_diff * z_b,
        0.75 * stretch_diff * z_b,
        stretch_diff * z_b,
    )
)


del data, fData, bcsData


# ====================================================================
# ====================  main function  ===============================
# ====================================================================


def boundary_warmup():  # This is run on the old data
    lamD = 500
    lamE = 1
    lamB = 10

    # define some network parameters
    N_residual = 1000000  # number of residual points in the domain
    N_b = 3000000  # number of boundary points
    num_gstep = 3000  # total number of training iterations
    batS = 10000  # batch size for each iteration
    lr = 1e-5  # initial learning rate

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
    t_train_b = t_b_warm[idx, :]
    x_train_b = x_b_warm[idx, :]
    y_train_b = y_b_warm[idx, :]
    z_train_b = z_b_warm[idx, :]

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
    model.saver.save(model.sess, save_transfer_models_to + "model_uv.ckpt")

    print("\n")
    print("Boundary warm-up done ......\n")


def data_warmup():  # This is data only
    lamD = 500
    lamE = 0
    lamB = 10

    # define some network parameters
    N_residual = 1000000  # number of residual points in the domain
    N_b = 500000  # max number of boundary points
    num_gstep = 10000  # total number of training iterations
    batS = 10000  # batch size for each iteration
    lr = 1e-3  # initial learning rate

    print("Size of MRI slices:", T_star.shape)

    # ====================================================================
    # downsample the residual points - no need to use all
    idx = np.random.choice(t_f.shape[0], np.uint32(N_residual), replace=False)
    print("Size of residual:", idx.shape)
    t_train_f = t_f_stretch[idx, :]
    x_train_f = x_f_stretch[idx, :]
    y_train_f = y_f_stretch[idx, :]
    z_train_f = z_f_stretch[idx, :]

    # ====================================================================
    # downsample the bc points - no need to use all if too many
    N_b = min([N_b, t_b.shape[0]])
    idx = np.random.choice(t_b.shape[0], np.uint32(N_b), replace=False)
    print("Size of BC:", idx.shape)
    t_train_b = t_b[idx, :]
    x_train_b = x_b[idx, :]
    y_train_b = y_b[idx, :]
    z_train_b = z_b_stretch[idx, :]

    # ====================================================================
    # ========================  training  ================================
    # ====================================================================
    model = NSFnets_TXYZ_BCS(
        T_star_stretch,
        X_star_stretch,
        Y_star_stretch,
        Z_star_stretch,
        U_star_stretch,
        V_star_stretch,
        W_star_stretch,
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

    model.saver.restore(model.sess, save_transfer_models_to + "model_uv.ckpt")
    model.evaluate_self()

    model.train(num_gstep=num_gstep, batch_size=batS, learning_rate=lr)
    model.evaluate_self()
    loss_log = model.loss_log
    nu_v_log = model.nu_v_log
    model.saver.save(model.sess, save_transfer_models_to + "model_uv.ckpt")

    print("\n")
    print("Data warm-up done ......\n")


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
    t_train_f = t_f_stretch[idx, :]
    x_train_f = x_f_stretch[idx, :]
    y_train_f = y_f_stretch[idx, :]
    z_train_f = z_f_stretch[idx, :]

    # ====================================================================
    # downsample the bc points - no need to use all if too many
    N_b = min([N_b, t_b.shape[0]])
    idx = np.random.choice(t_b.shape[0], np.uint32(N_b), replace=False)
    t_train_b = t_b[idx, :]
    x_train_b = x_b[idx, :]
    y_train_b = y_b[idx, :]
    z_train_b = z_b_stretch[idx, :]

    # ====================================================================
    # ========================  training  ================================
    # ====================================================================
    model = NSFnets_TXYZ_BCS(
        T_star_stretch,
        X_star_stretch,
        Y_star_stretch,
        Z_star_stretch,
        U_star_stretch,
        V_star_stretch,
        W_star_stretch,
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

    model.saver.restore(model.sess, save_transfer_models_to + "model_uv.ckpt")

    model.train(num_gstep=num_gstep, batch_size=batS, learning_rate=lr)
    model.evaluate_self()
    loss_log = model.loss_log
    nu_v_log = model.nu_v_log
    model.saver.save(model.sess, save_transfer_models_to + "model_uv.ckpt")

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
        boundary_warmup()

    graph2 = tf.Graph()
    with graph2.as_default():
        data_warmup()

    graph3 = tf.Graph()
    with graph3.as_default():
        error_u_3, error_v_3, error_w_3, error_mag_3 = PINN()

    print(" PINN: data + bcs + equ used: ")
    print("   relative l2 error u:    %e" % (error_u_3))
    print("   relative l2 error v:    %e" % (error_v_3))
    print("   relative l2 error w:    %e" % (error_w_3))
    print("   relative l2 error |V|:  %e" % (error_mag_3))
    print("---------------------------------------\n")
