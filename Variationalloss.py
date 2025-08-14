import torch
import math
import numpy as np

def total_loss(Q, P, A, R, train, Graphs, input):

    link_tensor = (Graphs.edge_index).T
    label = Graphs.label
    train_fea= Graphs.x[:,0:6].reshape(9,16,6)
    train_At = Graphs.x[:,7:27]
    label_Qt = label[:,0:20].reshape(9,16,20)
    label_Pt = label[:,20:40].reshape(9,16,20)
    label_At = train_At.reshape(9,16,20)

    mask = torch.where(label_Qt != 0, torch.tensor(1), torch.tensor(0))
    mask = torch.nonzero(mask.sum(dim=(1, 2)) != 0).squeeze()

    # De-normalize the network output
    Q = Q.reshape(9,16,20)
    Q = (Q[mask,:,:]*(10e-6-0)+0)
    P = P.reshape(9,16,20)
    P = (P[mask,:,:]*(18000-6000)+6000)
    A = A.reshape(9,16,20)
    A = (A[mask,:,:]*(5e-5-0)+0)
    label_Qt = label_Qt[mask,:,:]
    label_Pt = label_Pt[mask,:,:]
    label_At = label_At[mask,:,:]

    fea = train_fea[mask,:,:]
    parent_nodes = link_tensor[:, 0]
    child_nodes = link_tensor[:, 1]

    # Find the nodes among the child nodes that do not appear in the parent node, that is, the leaf nodes.
    leaf_nodes = child_nodes[~torch.isin(child_nodes, parent_nodes)]
    Bound_mark = ((leaf_nodes+1)/16-1).to(torch.long)
    if train == 1:
        Diff_Q = (Q - label_Qt)/label_Qt
        Diff_P = (P - label_Pt)/label_Pt
        Diff_A = (A - label_At)/label_At
        norm_Q = torch.norm(Diff_Q, p=2)
        norm_P = torch.norm(Diff_P, p=2)
        norm_A = torch.norm(Diff_A, p=2)
        #
        loss_mea = norm_Q + 1/1333*norm_P + norm_A

    loss_var = torch.tensor([0])

    # Calculate the variational loss
    vessel_number = label_Qt.size(0)
    x_number = label_Qt.size(1)
    t_number = label_Qt.size(2)
    MAP = input
    Reynoldnum = 600
    Vis = Graphs.x[0,5]
    rho = 1050
    h = 0.10* label_At
    W1 = label_At - math.pi*(R-h)^2
    W2 = label_At - math.pi*(R-h)^2
    W3 = label_At - math.pi*(R-h)^2
    W4 = label_At - math.pi*(R-h)^2
    W = torch.stack((W1, W2, W3, W4), dim=0)  # 形状: (4, 9, 16, 20)
    x_step = torch.mean(fea[mask,:,1],dim=1)
    for i in range(vessel_number):
        for j in range(x_number):
            for k in range(t_number):
                W1[i, j, k] = 1 + x_step[i] * j
                W2[i, j, k] = 1 + x_step[i] * j
                W3[i, j, k] = 1 + x_step[i] * j
    W1_dx = torch.ones(vessel_number, x_number, t_number)
    W2_dx = torch.ones(vessel_number, x_number, t_number)
    x_tensor = fea[mask,:,1]
    t_single = torch.linspace(0.04, 0.80, t_number)
    t_tensor = t_single.repeat(vessel_number, 1)
    x_tensor_diff = x_tensor[:, 1:] - x_tensor[:, :-1]
    delta_x = torch.mean(x_tensor_diff, dim=1, keepdim=True).unsqueeze(2).expand(-1, 16, 20)
    t_tensor_diff = t_tensor[:, 1:] - t_tensor[:, :-1]
    delta_t = torch.mean(t_tensor_diff, dim=1, keepdim=True).unsqueeze(2).expand(-1, 16, 20)
    a_tensor = fea[mask,:,2]
    Ax0 = a_tensor[:, :-1].unsqueeze(2).expand(-1, -1, 20)
    Ax1 = a_tensor[:, 1:].unsqueeze(2).expand(-1, -1, 20)
    # A
    A = torch.cat((A[:, :, :], A[:, :, 0].unsqueeze(2)), dim=2)
    Ax0t0 = A[:, :-1, :-1]
    Ax0t1 = A[:, :-1, 1:]
    Ax1t0 = A[:, 1:, :-1]
    Ax1t1 = A[:, 1:, 1:]
    A_t0 =  A[:, :, 0]  # 形状: (9, 16)
    At0 = A_t0.unsqueeze(2).repeat(1, 1, 20)
    # Q
    Q = torch.cat((Q[:, :, :], Q[:, :, 0].unsqueeze(2)), dim=2)
    Qx0t0 = Q[:, :-1, :-1]
    Qx0t1 = Q[:, :-1, 1:]
    Qx1t0 = Q[:, 1:, :-1]
    Qx1t1 = Q[:, 1:, 1:]

    # P
    P = torch.cat((P[:, :, :], P[:, :, 0].unsqueeze(2)), dim=2)
    Px0t0 = P[:, :-1, :-1]
    Px0t1 = P[:, :-1, 1:]
    Px1t0 = P[:, 1:, :-1]
    Px1t1 = P[:, 1:, 1:]

    Qin = Q[0, 0, :-1]
    Qout = Q[Bound_mark, -1, :-1]
    F = (4 / 3 * Q ** 2 / A + 1 / rho * A * P)
    Fin = F[0, 0, :-1]
    Fout = F[Bound_mark, -1, :-1]
    Stecheck = Ax1 < Ax0
    Ste = Stecheck.int()
    # Stenosis
    Kv = 32 * x_tensor_diff.unsqueeze(2).expand(-1, -1, 20) / 2 / torch.sqrt(Ax0 / torch.tensor(np.pi)) * (
            Ax0 / Ax1) ** 2
    Kt = 1.52
    Nt1 = Ax1 ** 2 * Qx0t0 ** 2 * (Kv / Reynoldnum + Kt / 2 * (Ax0 / Ax1 - 1))
    Nt2 = Ax0 ** 2 * Qx1t0 * delta_x
    N = Nt1 / Nt2
    G = Ste * N * (Qx0t0 / Ax0)
    K1 = 2* math.e**7
    K2 = -22.53
    K3 =  8.65* math.e**5
    Pext = 75*101325/760
    Loss1_1 = W1_dx * Qx0t1 * delta_x * delta_t
    Loss1_2 = W1 * (Ax0t1 - Ax0t0) * delta_x
    Loss1_3 = (W1[0, 0, :] * Qin - W1[ Bound_mark, -1, :] * Qout) * delta_t[0, 0, :]
    Loss1 = Loss1_1.sum(dim=(0, 1)) - Loss1_2.sum(dim=(0, 1)) + Loss1_3.sum(dim=(0))
    Loss2_1 = W2_dx * (4 / 3 * Qx0t0 ** 2 / Ax0t0 + 1 / rho * Ax0t0 * Px0t0) * delta_t * delta_x
    Loss2_2 = W2 * G * delta_t * delta_x
    Loss2_3 = W2 * (Qx0t1 - Qx0t0) * delta_x
    Loss2_4 = (W2[0, 0, :] * Fin - W2[ Bound_mark, -1, :] * Fout) * delta_t[0, 0, :]
    Loss2 = Loss2_1.sum(dim=(0, 1)) + Loss2_2.sum(dim=(0, 1)) - Loss2_3.sum(dim=(0, 1)) + Loss2_4.sum(dim=(0))
    MICCAI_variational_loss = torch.norm(Loss1, 2) + torch.norm(Loss2, 2)

    Loss_var_1 = W1*(Qx1t1 - Qx0t1) / delta_x + (Ax1t1 - Ax1t0) / delta_t  # old(Ax1t1-Ax0t1)
    Loss_var_2 = W2*(Qx1t1 - Qx1t0) / delta_t + 1 / rho * Ax0 * (Px1t1 - Px0t1) / delta_x - G + 0 * 4 / 3 * (
            Qx1t0 ** 2 / Ax1t0 - Qx0t0 ** 2 / Ax0t0) / delta_x
    Loss_var_3 = W3*(P - Pext -4/3/math.pi*(K1*math.e**(K2* torch.sqrt(Ax0 / torch.pi))+K3)*(1-At0/A))
    Loss_var_4 = W4(A-math.pi(R-h)**2)
    loss_var = torch.norm(Loss_var_1[:, :, :], 2) + torch.norm(Loss_var_2[:, :, :], 2) + torch.norm(Loss_var_3[:, :, :], 2) + torch.norm(Loss_var_4[:, :, :], 2)

    return MICCAI_variational_loss, loss_var, loss_mea
