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
    Setup603 = True

    torch.set_printoptions(precision=10)
    init_6p_file = "p_init.npy"
    init_Ts_file = "aT3s_init.npy"
    if Setup603:
        trackerdata_folder = "./calib_aT3_data/data_leika"
        robotdata_folder = "./calib_aT3_data/data_robot"
    else:
        trackerdata_folder = "./calib_aT3_data/data_ndi"
        robotdata_folder = "./calib_aT3_data/data_nachi"
    group_num = 6
    group_lists = ["group1", "group2", "group3", "group4", "group5", "group6"]
    num_sample_each = [20, 20, 20, 20, 20, 13]


    # marker position
    ini_p = np.load(init_6p_file)
    marker_p = Variable(torch.tensor(ini_p.squeeze().tolist(), requires_grad=True).cuda())


    # d  a  alpha  theta
    DH_link_4 = Variable(torch.tensor([0.2013, 0., -0.5 * np.pi, -0.5 * np.pi], requires_grad=True).cuda())
    DH_link_5 = Variable(torch.tensor([0.1025, 0., 0.5 * np.pi, 0.], requires_grad=True).cuda())
    DH_link_6 = Variable(torch.tensor([0.094, 0., 0., 0.], requires_grad=True).cuda())


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
        file_path = os.path.join(trackerdata_folder, group + ".csv")
        tracker_data = pd.read_csv(file_path, sep = ';', encoding='unicode_escape')
        X = tracker_data['X  [mm]'].to_list()
        Y = tracker_data['Y  [mm]'].to_list()
        Z = tracker_data['Z  [mm]'].to_list()

        marker_point_group = Variable(torch.from_numpy(np.array(
            [[X_i / 1000.0, Y_i / 1000.0, Z_i / 1000.0] for X_i, Y_i, Z_i in
             zip(X[:num_sample_each[i]],
                 Y[:num_sample_each[i]],
                 Z[:num_sample_each[i]])])).cuda())

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








