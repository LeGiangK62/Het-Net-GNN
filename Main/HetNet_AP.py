import torch
import numpy as np
import h5py
from scipy.spatial import distance_matrix

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader


# from WSN_GNN import generate_channels_wsn
# from hgt_conv import HGTGNN
from .GNN_AP import RGCN


def generate_channels(num_ap, num_user, num_samples, var_noise=1.0, radius=1):
    # print("Generating Data for training and testing")

    # if num_ap != 1:
    #     raise Exception("Can not generate data for training and testing with more than 1 base station")
    # generate position
    dist_mat = []
    position = []
    index_user = np.tile(np.arange(num_user), (num_ap, 1))
    index_ap = np.tile(np.arange(num_ap).reshape(-1, 1), (1, num_user))

    index = np.array([index_user, index_ap])

    # Calculate channel
    CH = 1 / np.sqrt(2) * (np.random.randn(num_samples, num_ap, num_user)
                           + 1j * np.random.randn(num_samples, num_ap, num_user))
    CH = CH ** 2

    if radius == 0:
        Hs = abs(CH * np.ones((num_samples, num_ap, num_user)))
    else:
        for each_sample in range(num_samples):
            pos = []
            pos_BS = []

            for i in range(num_ap):
                r = radius * (np.random.rand())
                theta = np.random.rand() * 2 * np.pi
                pos_BS.append([r * np.sin(theta), r * np.cos(theta)])
                pos.append([r * np.sin(theta), r * np.cos(theta)])
            pos_user = []

            for i in range(num_user):
                r = 0.5 * radius + 0.5 * radius * np.random.rand()
                theta = np.random.rand() * 2 * np.pi
                pos_user.append([r * np.sin(theta), r * np.cos(theta)])
                pos.append([r * np.sin(theta), r * np.cos(theta)])

            pos = np.array(pos)
            pos_BS = np.array(pos_BS)
            dist_matrix = distance_matrix(pos_BS, pos_user)
            # dist_matrixp = distance_matrix(pos[1:], pos[1:])
            dist_mat.append(dist_matrix)
            position.append(pos)

        dist_mat = np.array(dist_mat)
        position = np.array(position)

        # Calculate Free space pathloss
        # f = 2e9
        # c = 3e8
        # FSPL_old = 1 / ((4 * np.pi * f * dist_mat / c) ** 2)
        FSPL = - (120.9 + 37.6 * np.log10(dist_mat / 1000))
        FSPL = 10 ** (FSPL / 10)
        # FSPL = np.sqrt(FSPL)
        # print(f'FSPL_old:{FSPL_old.sum()}')
        # print(f'FSPL_new:{FSPL.sum()}')
        Hs = abs(CH * FSPL)

    adj = adj_matrix(num_user, num_ap)
    noise = var_noise

    return Hs / var_noise, 1, position, adj, index


# region Create HeteroData from the wireless system
def convert_to_hetero_data(channel_matrices, p_max, ap_selection_matrix):
    graph_list = []
    num_sam, num_aps, num_users = channel_matrices.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(num_sam):
        x1 = torch.ones(num_users, 1) * p_max
        x2 = torch.zeros(num_users, 1)  # power allocation
        # x3 = torch.tensor(ap_selection)
        # user_feat = torch.cat((x1,x2,x3),1)  # features of user_node
        user_feat = torch.cat((x1, x2), 1)  # features of user_node
        ap_feat = torch.ones(num_aps, 2)  # features of user_node
        y1 = channel_matrices[i, :, :].reshape(-1, 1)
        # y2 = ap_selection_matrix[i, :, :].reshape(-1, 1)
        y2 = torch.ones(num_users * num_aps, 1)

        edge_feat_uplink = np.concatenate((y1, y2), 1)

        edge_feat_downlink = np.concatenate((y1, y2), 1)
        graph = HeteroData({
            'ue': {'x': user_feat.to(device)},
            'ap': {'x': ap_feat.to(device)}
        })
        # Create edge types and building the graph connectivity:
        graph['ue', 'uplink', 'ap'].edge_attr = torch.tensor(edge_feat_uplink, dtype=torch.float32).to(device)
        graph['ap', 'downlink', 'ue'].edge_attr = torch.tensor(edge_feat_downlink, dtype=torch.float32).to(device)

        graph['ue', 'uplink', 'ap'].edge_index = torch.tensor(adj_matrix(num_users, num_aps).transpose(),
                                                              dtype=torch.int64, device=device).contiguous()
        graph['ap', 'downlink', 'ue'].edge_index = torch.tensor(adj_matrix(num_aps, num_users).transpose(),
                                                                dtype=torch.int64, device=device).contiguous()
        graph_list.append(graph)
    return graph_list


def adj_matrix(num_from, num_dest):
    adj = []
    for i in range(num_from):
        for j in range(num_dest):
            adj.append([i, j])
    return np.array(adj)


#region Training and Testing functions
def loss_function(output, batch, noise_matrix, size, p_cir, is_train=True, is_log=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_ue, num_ap, batch_size = size

    output = torch.reshape(output, (batch_size, num_ue, -1))
    channel_matrix = batch['ue', 'ap']['edge_attr'][:,0]
    power_max = output[:, :, 0]
    power = output[:, :, 1] * power_max
    ap_selection = batch['ue', 'ap']['edge_attr'][:, 1]
    P = torch.reshape(ap_selection, (-1, num_ap, num_ue))
    # P = torch.softmax(P, dim=1)
    # # P = torch.round(P)
    # max_indices = torch.argmax(P, dim=1)

    # # Create a new tensor where the maximum value is set to 1 and the rest to 0
    # result = torch.zeros_like(P)
    # result.scatter_(1, max_indices.unsqueeze(1), 1)
    # P = result

    # print(P.shape)
    # print(f'{(P == 1).sum().item()}/{len(P)}')


    G = torch.reshape(channel_matrix, (-1, num_ap, num_ue))
    power = power.unsqueeze(1)
    # P = P * power
    sum_rate, rate = sum_rate_calculation(P * power, P, G, noise_matrix)
    power_all = torch.sum(power, 1).unsqueeze(-1)

    # ee = torch.mean( torch.div(rate, power_all + p_cir)) # Personal Energy efficiency
    ee = torch.mean( torch.div(rate, power_all + p_cir)) # Option 1
    sum_rate_batch = torch.sum(rate, dim=1)
    sum_power_batch = torch.sum(power_all + p_cir, dim=1)
    ee_batch = torch.div(sum_rate_batch, sum_power_batch)
    ee_mean = torch.mean(ee_batch)

    if is_log:
        print(f'power: {P[0]}')
        # print(f'channel: {G[0]}')
        # print(f'desired_signal: {desired_signal[0]}')
        # print(f'interference: {interference[0]}')

    if is_train:
        # return sum_rate, torch.neg(sum_rate), torch.mean(sum_power_batch)
        return sum_rate, torch.neg(ee_mean), torch.mean(sum_power_batch)
        # return sum_rate, torch.neg(torch.mean(sum_rate_batch)) #/ mean_power

    else:
        # return sum_rate / mean_power
        return torch.neg(ee_mean)
        #  return torch.neg(sum_rate)
        # return torch.neg(torch.mean(sum_rate_batch)) #/ mean_power




def get_sum_rate(output, batch, noise_matrix, size, is_train=True, is_log=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_ue, num_ap, batch_size = size

    output = torch.reshape(output, (batch_size, num_ue, -1))
    channel_matrix = batch['ue', 'ap']['edge_attr'][:,0]
    power_max = output[:, :, 0]
    power = output[:, :, 1] * power_max
    ap_selection = batch['ue', 'ap']['edge_attr'][:, 1]
    P = torch.reshape(ap_selection, (-1, num_ap, num_ue))

    G = torch.reshape(channel_matrix, (-1, num_ap, num_ue))
    power = power.unsqueeze(1)
    P = P * power
    sum_rate, _ = sum_rate_calculation(P, G, noise_matrix)
    return sum_rate



def train(data_loader, noise, p_cir):
    model.train()
    device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_examples = total_loss = sumRate = sumPower = 0
    for batch in data_loader:
        optimizer.zero_grad()
        batch = batch.to(device_type)
        #
        num_ues = batch['ue'].num_nodes
        num_aps = batch['ap'].num_nodes
        num_edges = batch['ue', 'ap'].num_edges
        batch_size = int(num_ues * num_aps / num_edges)
        num_ues = int(num_ues / batch_size)
        num_aps = int(num_aps / batch_size)
        #
        out, edge = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        out = out['ue']
        tmp_sumRate, tmp_loss, tmp_sumPower = loss_function(out, batch, noise, (num_ues, num_aps, batch_size), p_cir, True)
        tmp_loss.backward()
        optimizer.step()
        total_examples += batch_size
        total_loss += float(tmp_loss) * batch_size
        sumRate += float(tmp_sumRate) * batch_size
        sumPower += float(tmp_sumPower) * batch_size

    return sumRate / total_examples, total_loss / total_examples, sumPower / total_examples


def test(data_loader, noise, p_cir, is_log=False):
    model.eval()
    device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_examples = total_loss = 0
    for batch in data_loader:
        batch = batch.to(device_type)
        #
        num_ues = batch['ue'].num_nodes
        num_aps = batch['ap'].num_nodes
        num_edges = batch['ue', 'ap'].num_edges
        batch_size = int(num_ues * num_aps / num_edges)
        num_ues = int(num_ues / batch_size)
        num_aps = int(num_aps / batch_size)
        #
        out, edge = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        out = out['ue']
        tmp_loss = loss_function(out, batch, noise, (num_ues, num_aps, batch_size), p_cir, False)
        total_examples += batch_size
        total_loss += float(tmp_loss) * batch_size

    last_batch_edge = (edge['ue','uplink','ap'])

    if is_log:
      # print(out[:3])
      tensor = (edge['ue','uplink','ap'])
      # print(tensor)
      print(f'The number of activated link: {(tensor[:,1] == 0).sum().item()}/{len(tensor[:,1])}' )
      # tmp_loss = loss_function(out, batch, noise, (num_ues, num_aps, batch_size), False, True)
    return total_loss / total_examples, last_batch_edge


def sum_rate_calculation(power_matrix, ap_selection, channel_matrix,  noise_matrix):
    P = power_matrix
    G = channel_matrix
    new_noise = noise_matrix
    desired_signal = torch.sum(torch.mul(P, G), dim=1).unsqueeze(-1)
    P_trans = P.permute(0,2,1)
    P_UE = torch.sum(P_trans, dim=2).unsqueeze(-1)  # P_UE[n] = The power n-th UE transmits
    all_received_signal = torch.matmul(G, P_UE)
    all_signal = torch.matmul(ap_selection.permute(0,2,1), all_received_signal)
    # max_P,_ = torch.max(P_trans, dim=2)
    # max_P = max_P.unsqueeze(-1)
    # print(max_P)
    # all_signal = torch.div(tmp, max_P)
    interference = -desired_signal + all_signal + noise_matrix
    rate = torch.log(1 + torch.div(desired_signal, interference))
    sum_rate = torch.mean(torch.sum(rate, 1))
    return sum_rate, rate


# def sum_rate_calculation(power_matrix, channel_matrix,  noise_matrix):
#     P = power_matrix
#     G = channel_matrix
#     new_noise = noise_matrix
#     desired_signal = torch.sum(torch.mul(P, G), dim=1).unsqueeze(-1)
#     G_UE = torch.sum(G, dim=2).unsqueeze(-1)
#     all_signal = torch.matmul(P.permute((0, 2, 1)), G_UE)
#     interference = all_signal - desired_signal + new_noise
#     rate = torch.log(1 + torch.div(desired_signal, interference))
#     return torch.mean(torch.sum(rate, 1))
#endregion
