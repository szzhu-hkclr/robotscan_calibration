import numpy as np
import torch
import torch.nn as nn
import roma
import pandas as pd
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os

def ParameterStack(params, keys=None, is_param=None, fill=0):
    if keys is not None:
        params = [params[k] for k in keys]

    if fill > 0:
        params = [_ravel_hw(p, fill) for p in params]

    requires_grad = params[0].requires_grad
    assert all(p.requires_grad == requires_grad for p in params)

    params = torch.stack(list(params)).float().detach()
    if is_param or requires_grad:
        params = nn.Parameter(params)
        params.requires_grad_(requires_grad)
    return params

def _ravel_hw(tensor, fill=0):
    # ravel H,W
    tensor = tensor.view((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

    if len(tensor) < fill:
        tensor = torch.cat((tensor, tensor.new_zeros((fill - len(tensor),)+tensor.shape[1:])))
    return tensor


def from_dh(d, a, alpha, theta):

    from torch import sin as s, cos as c

    T = torch.zeros((4, 4), device=theta.device, dtype=theta.dtype)

    T[0, 0] = c(theta)
    T[0, 1] = -s(theta) * c(alpha)
    T[0, 2] = s(theta) * s(alpha)
    T[0, 3] = a * c(theta)
    T[1, 0] = s(theta)
    T[1, 1] = c(theta) * c(alpha)
    T[1, 2] = -c(theta) * s(alpha)
    T[1, 3] = a * s(theta)
    T[2, 1] = s(alpha)
    T[2, 2] = c(alpha)
    T[2, 3] = d
    T[3, 3] = 1.

    return T



## kinematic model
def kinematic_forward_eachgroup(pose_group_i, DH_link_4, DH_link_5, DH_link_6, marker_p, joint_states_group_i):

    num_points = joint_states_group_i.shape[0]

    a_T_3 = roma.RigidUnitQuat(linear=pose_group_i[0:4], translation=pose_group_i[4:7]).to_homogeneous()

    a_ps = []
    for j in range(num_points):
        joint_state = joint_states_group_i[j]

        three_T_four = from_dh(DH_link_4[0], DH_link_4[1], DH_link_4[2], DH_link_4[3] + joint_state[3]).type_as(marker_p)
        four_T_five = from_dh(DH_link_5[0], DH_link_5[1], DH_link_5[2], DH_link_5[3] + joint_state[4]).type_as(marker_p)
        five_T_six = from_dh(DH_link_6[0], DH_link_6[1], DH_link_6[2], DH_link_6[3] + joint_state[5]).type_as(marker_p)

        a_p = a_T_3 @ three_T_four @ four_T_five @ five_T_six @ torch.cat((marker_p, torch.tensor([1.0]).cuda()), dim=0)
        a_ps.append(a_p[:3])

    a_ps = torch.stack(a_ps)

    return a_ps




if __name__ == '__main__':
    Setup603 = False
    OnlyGroup3n4 = False

    torch.set_printoptions(precision=10)
    init_6p_file = "p_init.npy"
    init_Ts_file = "aT3s_init.npy"

    group_lists = []

    if Setup603:
        trackerdata_folder = "./calib_aT3_data/data_leika"
        robotdata_folder = "./calib_aT3_data/data_robot"
        group_lists = ["group1", "group2", "group3", "group4", "group5", "group6"]
        # d  a  alpha  theta
        DH_link_4 = Variable(torch.tensor([0.2013, 0., -0.5 * np.pi, -0.5 * np.pi], requires_grad=True).cuda())
        DH_link_5 = Variable(torch.tensor([0.1025, 0., 0.5 * np.pi, 0.], requires_grad=True).cuda())
        DH_link_6 = Variable(torch.tensor([0.094, 0., 0., 0.], requires_grad=True).cuda())
        num_sample_each = [20, 20, 20, 20, 20, 13]
    else:
        trackerdata_folder = "./calib_aT3_data/data_ndi"
        robotdata_folder = "./calib_aT3_data/data_nachi"
        if OnlyGroup3n4:
            group_lists = [ "group3", "group4"]
        else:
            group_lists = ["group1", "group2", "group3", "group4", "group5", "group6"]
        # d  a  alpha  theta
        DH_link_4 = Variable(torch.tensor([0.810, 0., -0.5 * np.pi, 0.00], requires_grad=True).cuda())
        DH_link_5 = Variable(torch.tensor([0.0, 0., 0.5 * np.pi, 0.], requires_grad=True).cuda())
        DH_link_6 = Variable(torch.tensor([0.115, 0., 0., 0.], requires_grad=True).cuda())
        num_sample_each = [20, 20, 20, 20, 20, 20]
    
    group_num = len(group_lists)


    # marker position
    ini_p = np.load(init_6p_file)
    marker_p = Variable(torch.tensor(ini_p.squeeze().tolist(), requires_grad=True).cuda())




    # poses of joint_3 in each group
    pose_all_groups = nn.ParameterList([nn.Parameter(torch.randn(7).cuda()) for _ in range(group_num)])    # (tx, ty, tz, w), (x, y, z)
    pose_all_groups = ParameterStack(pose_all_groups, is_param=True)


    ################# read the computed poses a.nd initialize the pose_all_groups
    ini_Ts = np.load(init_Ts_file)
    init_poses = []
    for i in range(group_num):
        init_T_i = ini_Ts[i]
        init_R_i = torch.from_numpy(init_T_i[:3, :3]).cuda()
        init_t_i = torch.from_numpy(init_T_i[:3, 3]).cuda()

        with torch.no_grad():
            pose_all_groups[i][0:4] = roma.rotmat_to_unitquat(init_R_i)
            pose_all_groups[i][4:7] = init_t_i


    ### measurements
    # tracker data
    marker_points = []
    for group, i in zip(group_lists, range(group_num)):
        if Setup603:
            file_path = os.path.join(trackerdata_folder, group + ".csv")
            tracker_data = pd.read_csv(file_path, sep = ',', encoding='unicode_escape')
            X = tracker_data['X  [mm]'].to_list()
            Y = tracker_data['Y  [mm]'].to_list()
            Z = tracker_data['Z  [mm]'].to_list()

            marker_point_group = Variable(torch.from_numpy(np.array(
                [[X_i / 1000.0, Y_i / 1000.0, Z_i / 1000.0] for X_i, Y_i, Z_i in
                zip(X[:num_sample_each[i]],
                    Y[:num_sample_each[i]],
                    Z[:num_sample_each[i]])])).cuda())
            marker_points.append(marker_point_group.float())
        else:
            file_path = os.path.join(trackerdata_folder, group + ".txt")
            with open(file_path, "r") as f:
                lines = f.readlines()
                # Parse the lines, skipping empty lines
                parsed_lines = [
                    [float(value.replace(",", "")) for value in line.split(",")[:3]]
                    for line in lines if line.strip() != ""
                ]
                # Extract X, Y, Z coordinates and normalize
                X = [line[0] / 1000.0 for line in parsed_lines]
                Y = [line[1] / 1000.0 for line in parsed_lines]
                Z = [line[2] / 1000.0 for line in parsed_lines]

                # Create marker_point_group for this group
                marker_point_group = Variable(torch.from_numpy(np.array(
                    [[X_i, Y_i, Z_i] for X_i, Y_i, Z_i in
                    zip(X[:num_sample_each[i]],
                        Y[:num_sample_each[i]],
                        Z[:num_sample_each[i]])]
                ))).cuda()

                marker_points.append(marker_point_group.float())


    # joint states
    robot_states = []
    for group, i in zip(group_lists, range(group_num)):
        file_path = os.path.join(robotdata_folder, group + ".txt")
        robot_states_group = []

        with open(file_path, "r") as f:
            for line in f.readlines():
                if line != "\n":
                    line = line.strip('\n')

                    line_data = line.split(' ')
                    state = []
                    for data in line_data:
                        data = data.replace(",", "")
                        state.append(float(data))

                    robot_states_group.append(state)

        robot_state_group = Variable(torch.from_numpy(np.array(robot_states_group[:num_sample_each[i]])).cuda())
        robot_states.append(robot_state_group)

    DH_link_4.requires_grad = True
    DH_link_5.requires_grad = True
    DH_link_6.requires_grad = True
    marker_p.requires_grad = True
    pose_all_groups.requires_grad = True
    optimizer = torch.optim.Adam([DH_link_4, DH_link_5, DH_link_6, marker_p, pose_all_groups], lr=1e-4)
    lossfun = nn.MSELoss(reduction='sum')

    num_epochs = 500
    # num_epochs = 20000
    epoch_list = []
    cost_list = []
    for epoch in range(num_epochs):

        total_loss = 0
        for idx in range(group_num):

            kine_markers = kinematic_forward_eachgroup(pose_all_groups[idx], DH_link_4, DH_link_5, DH_link_6, marker_p, robot_states[idx])
            loss = lossfun(kine_markers, marker_points[idx])
            total_loss += loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # 120 items = 20 samples *6 grps
        cost = np.sqrt(total_loss.item()/120.0) * 1000
        print('Epoch [{}/{}], Loss: {:.8f}'.format(epoch + 1, num_epochs, cost))

        epoch_list.append(epoch)
        cost_list.append(cost)
        print("DH_link_4:", DH_link_4.detach().cpu().numpy())


    plt.plot(epoch_list, cost_list)
    plt.ylabel('Average Cost (mm)')
    plt.xlabel('Iteration Number')
    plt.show()

    aT3s = []
    for idx in range(group_num):
        pose_idx = pose_all_groups[idx]
        aT3_idx = roma.RigidUnitQuat(linear=pose_idx[0:4], translation=pose_idx[4:7]).to_homogeneous().detach().cpu().numpy()
        aT3s.append(aT3_idx)

    np.save('./aT3s_opt.npy', aT3s)

    print("6p: ", marker_p.detach().cpu().numpy())
    for idx in range(group_num):
        print("aT%s:"%(idx+1), aT3s[idx])
    print("DH_link_4:", DH_link_4.detach().cpu().numpy())
    print("DH_link_5:", DH_link_5.detach().cpu().numpy())
    print("DH_link_6:", DH_link_6.detach().cpu().numpy())

# 16w 2025-01-17
# Epoch [5000/5000], Loss: 0.23856669
# DH_link_4: [ 8.1010616e-01 -3.8129519e-04 -1.5701258e+00  2.0415702e-05]
# QStandardPaths: XDG_RUNTIME_DIR not set, defaulting to '/tmp/runtime-hkclr'
# 6p:  [ 0.00373765 -0.06330439  0.11231989]
# aT1: [[-9.4220603e-01  2.6518956e-02  3.3445978e-01 -3.2658494e-01]
#  [ 3.3550948e-01  7.4830934e-02  9.3922973e-01 -9.4654363e-01]
#  [-1.2053549e-04  9.9700344e-01 -7.9390898e-02 -1.9397105e+00]
#  [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  1.0000000e+00]]
# aT2: [[-0.95194954  0.03829085  0.3011458  -0.23048654]
#  [ 0.2509338  -0.4585714   0.8515324  -1.1210781 ]
#  [ 0.17084265  0.8869098   0.42727825 -2.0627275 ]
#  [ 0.          0.          0.          1.        ]]
# aT3: [[-0.94418716  0.02521627  0.33116978 -0.30009001]
#  [ 0.33192652  0.10643279  0.9382407  -1.0089245 ]
#  [-0.01157801  0.99490446 -0.10876466 -1.9319137 ]
#  [ 0.          0.          0.          1.        ]]
# aT4: [[-0.88580084  0.03879194  0.46181768 -0.3690665 ]
#  [ 0.3853299  -0.49184477  0.78040564 -0.64096147]
#  [ 0.25749034  0.8694868   0.42085016 -1.7969393 ]
#  [ 0.          0.          0.          1.        ]]
# aT5: [[-0.9179961   0.03660384  0.3942349  -0.37511262]
#  [ 0.3672285  -0.2933903   0.8823509  -0.7005891 ]
#  [ 0.14800078  0.9550184   0.25595602 -1.8807119 ]
#  [ 0.          0.          0.          1.        ]]
# aT6: [[-0.91314626  0.03001811  0.40607482 -0.3745082 ]
#  [ 0.40366358 -0.06407367  0.9124605  -0.6983977 ]
#  [ 0.05341883  0.9973101   0.0463999  -1.9289974 ]
#  [ 0.          0.          0.          1.        ]]
# DH_link_4: [ 8.1010616e-01 -3.8129519e-04 -1.5701258e+00  2.0415702e-05]
# DH_link_5: [1.26107159e-04 1.17182695e-04 1.57024145e+00 5.60347806e-04]
# DH_link_6: [ 1.1500830e-01  4.6946361e-06 -3.3456785e-05  9.3496355e-06]




# 16w 2025-02-24
# Epoch [34998/35000], Loss: 0.19597493
# DH_link_4: [ 4.1023743e-01  8.8541874e-06 -1.5701793e+00 -3.6477977e-01]
# Epoch [34999/35000], Loss: 0.19598071
# DH_link_4: [ 4.1023681e-01  8.9592741e-06 -1.5701792e+00 -3.6477965e-01]
# Epoch [35000/35000], Loss: 0.19598491
# DH_link_4: [ 4.1023746e-01  8.8334800e-06 -1.5701793e+00 -3.6477974e-01]
# QStandardPaths: XDG_RUNTIME_DIR not set, defaulting to '/tmp/runtime-hkclr'
# 6p:  [ 0.01670381 -0.08515751 -0.04159189]
# aT1: [[-0.9398684   0.6448092   0.33975202 -0.3782209 ]
#  [ 0.23142695 -0.2617042   1.1368884  -1.013092  ]
#  [ 0.69112366  0.96451795  0.08133939 -1.8212687 ]
#  [ 0.          0.          0.          1.        ]]
# aT2: [[-0.86554277  0.5969358   0.55829656 -0.4887667 ]
#  [ 0.7442462   0.23951815  0.8977307  -0.5959727 ]
#  [ 0.33782393  1.0017437  -0.5473356  -1.965859  ]
#  [ 0.          0.          0.          1.        ]]
# aT3: [[-0.84270126  0.61004686  0.57860416 -0.47409144]
#  [ 0.6274537  -0.0890171   1.007702   -0.24384812]
#  [ 0.55968016  1.018334   -0.25853306 -1.8820455 ]
#  [ 0.          0.          0.          1.        ]]
# aT4: [[-0.7932567   0.5803451   0.67060655 -0.3624158 ]
#  [ 0.45571536 -0.50511605  0.9761921  -0.14120695]
#  [ 0.7608134   0.9076493   0.11447936 -1.5893508 ]
#  [ 0.          0.          0.          1.        ]]
# aT5: [[-0.7282442   0.51800936  0.7855213  -0.37147802]
#  [ 0.30642292 -0.8085983   0.81730705 -0.4522226 ]
#  [ 0.889653    0.7025327   0.3615002  -1.470292  ]
#  [ 0.          0.          0.          1.        ]]
# aT6: [[-0.8630558   0.5656917   0.5912837  -0.41835424]
#  [ 0.12287053 -0.7600423   0.90649176 -0.86203337]
#  [ 0.809028    0.7189001   0.49309745 -1.7299995 ]
#  [ 0.          0.          0.          1.        ]]
# DH_link_4: [ 4.1023746e-01  8.8334800e-06 -1.5701793e+00 -3.6477974e-01]
# DH_link_5: [-2.0575602e-05 -1.7802379e-04  1.5701313e+00  1.0053637e-03]
# DH_link_6: [0.26512548 0.04879135 0.01211041 0.04397908]