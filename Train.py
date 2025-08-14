import torch
import time
import os
import scipy.io as sio
import numpy as np
import json
import math
import torch.optim as optim
import torch.nn as nn
from VarNN import VarNN
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Variationalloss import total_loss

def Data_progress(program_root):

    # Input data: geometry of vessel
    graphs = []
    feature_data_path = os.path.join(program_root, "Data.mat")
    link_data_path = os.path.join(program_root, "Link.mat")
    Amat_cell_path = os.path.join(program_root, "Acell.mat")
    Qmat_cell_path = os.path.join(program_root, "Qcell.mat")
    Pmat_cell_path = os.path.join(program_root, "Pcell.mat")
    Time_path = os.path.join(program_root, "Time.mat")
    Inlet_boundary_path = os.path.join(program_root, "Inlet.mat")

    # 读取MAT文件
    feature_data = sio.loadmat(feature_data_path)
    link_data = sio.loadmat(link_data_path)
    Amat_cell = sio.loadmat(Amat_cell_path)
    Qmat_cell = sio.loadmat(Qmat_cell_path)
    Pmat_cell = sio.loadmat(Pmat_cell_path)
    Time = sio.loadmat(Time_path)
    Inlet_boundary = sio.loadmat(Inlet_boundary_path)

    # 将NumPy数组转换为Tensor‘
    Time_list   = Time['Time']
    At_data = Amat_cell['Amat']
    Qt_data = Qmat_cell['Qmat']
    Pt_data = Pmat_cell['Pmat']
    x       = feature_data['data_cell']
    link    = link_data['link_cell']
    Inlet   = Inlet_boundary['Inlet']
    max_node = 144 # Coronary segmentation node
    for i in range(len(x)):

        x_i = np.vstack(x[i])
        A_i = np.vstack(At_data[i])
        Q_i = np.vstack(Qt_data[i])
        P_i = np.vstack(Pt_data[i])
        link_i = np.vstack(link[i])
        tensor_x_i = torch.tensor(x_i, dtype=torch.float32)
        tensor_P_i = torch.tensor(P_i, dtype=torch.float32)
        tensor_Q_i = torch.tensor(Q_i, dtype=torch.float32)

        # A_ij_DES
        tensor_A_i = torch.tensor(A_i, dtype=torch.float32)
        # Q_mean_DES
        tensor_Qmean_i = torch.tensor(np.sum(Q_i, axis=1), dtype=torch.float32).unsqueeze(1)  # 变成 [112, 1]
        tensor_Qmean_i = tensor_Qmean_i.expand(-1, 20)  # 变成 [112, 20]
        # P_in_DES
        tensor_inlet_i  = np.zeros((x_i.shape[0], 20))
        tensor_inlet_i[0,:] = Inlet
        tensor_inlet_i = torch.tensor(tensor_inlet_i)
        tensor_link_i = torch.tensor(link_i, dtype=torch.float32)
        tensor_link_i = tensor_link_i.long() - 1

        # PyTorch Geometric
        edge_index = tensor_link_i.t().contiguous()
        num_edges = edge_index.size(1)
        padding_needed = max_node - 1 - num_edges
        node_features = torch.cat((tensor_x_i, tensor_inlet_i, tensor_Qmean_i, tensor_A_i), dim=1)

        Y = torch.cat((tensor_Q_i, tensor_P_i), dim=1)
        # 如果边的数量小于 144，进行填充
        if padding_needed > 0:
            # 在第二个维度进行填充，后面填充 zeros
            edge_index = F.pad(edge_index, (0, padding_needed), "constant", 0)

        num_nodes = node_features.size(0)
        padding_needed = 144 - num_nodes
        if padding_needed > 0:
            # 在第一维度进行填充，在开头和末尾都不填充，仅在后面填充
            node_features = F.pad(node_features, (0, 0, 0, padding_needed), "constant", 0)
            Y = F.pad(Y, (0, 0, 0, padding_needed), "constant", 0)

        graph           = Data(x=node_features, edge_index=edge_index, label=Y)
        graphs.append(graph)

    return graphs

def train():

    device = torch.device("cuda"
                          if torch.cuda.is_available()
                          else "cpu")

    program_root = r"D:\ZQ\TMI\Code"

    Data_train = Data_progress(program_root)
    Data_training = DataLoader(Data_train, batch_size=1, shuffle=True)

    model = VarNN(
        node_num=144, out_len=12, edge_index=144, data_dim=3, seg_len=4,
        win_size=4, factor=10, d_model=128, d_ff=256, n_heads=4, e_layers=2, dropout=0.1
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
        betas=(0.95, 0.99),
        eps=1e-6,
        weight_decay=1e-5
    )

    epochs = 50000
    accumulation = 8
    tic = time.time()
    writer = SummaryWriter('TensorBoard/PVDPM-train')
    best_loss, best_model_dict, best_epoch = float('inf'), None, 0
    update_step = 0

    for istep in range(epochs):
        model.train()
        optimizer.zero_grad()
        epoch_loss = np.zeros(shape=6, dtype=np.float32)

        for i_batch, data in enumerate(Data_training):
            Q_mat, P_mat, A_mat, R_mat = model(data.to(device))

            loss_item = total_loss(Q_mat, P_mat, A_mat, R_mat, data)
            loss = loss_item[0] / accumulation
            loss.backward()
            epoch_loss += torch.stack(loss_item).detach().cpu().numpy()

            if (i_batch + 1) % accumulation == 0 or i_batch == len(Data_training) - 1:
                epoch_loss /= accumulation

                if epoch_loss[0] < best_loss:
                    best_loss = epoch_loss[0]
                    best_model_dict = model.state_dict()
                    best_epoch = istep


                nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()

                print_str = (
                    f'Train step: {update_step} \tLoss: {epoch_loss[0]:.10f}\tNS_nor: {epoch_loss[1]:.10f}'
                    f'\tJoint_nor: {epoch_loss[2]:.10f}\tNS: {epoch_loss[3]:.10f}'
                    f'\tFlux: {epoch_loss[4]:.10f}\tPressure: {epoch_loss[5]:.10f}'
                )
                print(print_str)

                # TensorBoard
                writer.add_scalar('loss/tol_loss', epoch_loss[0], update_step)
                writer.add_scalar('loss/loss_segment', epoch_loss[1], update_step)
                writer.add_scalar('loss/loss_joint', epoch_loss[2], update_step)

                update_step += 1
                epoch_loss = np.zeros(shape=6, dtype=np.float32)

                if update_step % 1000 == 0:
                    torch.save(model.state_dict(), f"SaveDir/step{update_step}_model.pt")

    torch.save(model.state_dict(), f"SaveDir/epoch{epochs}_model.pt")
    torch.save(best_model_dict, f"SaveDir/best_epoch{best_epoch}_model.pt")

    toc = time.time()
    print("elapse time in parallel = ", toc - tic)


train()
