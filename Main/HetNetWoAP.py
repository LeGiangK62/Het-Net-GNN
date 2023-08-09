import torch
import numpy as np
from scipy.spatial import distance_matrix

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader


from WSN_GNN import generate_channels_wsn
from hgt_conv import HGTGNN
from GNNWoAP import RGCN


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
    CH = 1 / np.sqrt(2) * (np.random.randn(num_samples, 1, num_user)
                           + 1j * np.random.randn(num_samples, 1, num_user))

    if radius == 0:
        Hs = abs(CH)
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
        FSPL = - (120.9 + 37.6 * np.log10(dist_mat/1000))
        FSPL = 10 ** (FSPL / 10)

        # print(f'FSPL_old:{FSPL_old.sum()}')
        # print(f'FSPL_new:{FSPL.sum()}')
        Hs = abs(CH * FSPL)

    adj = adj_matrix(num_user, num_ap)
    noise = var_noise

    return Hs, noise, position, adj, index

#region Create HeteroData from the wireless system
def convert_to_hetero_data(channel_matrices, p_max, ap_selection_matrix):
    graph_list = []
    num_sam, num_aps, num_users = channel_matrices.shape
    for i in range(num_sam):
        x1 = torch.ones(num_users, 1) * p_max
        x2 = torch.zeros(num_users, 1)  # power allocation
        # x3 = torch.tensor(ap_selection)
        # user_feat = torch.cat((x1,x2,x3),1)  # features of user_node
        user_feat = torch.cat((x1, x2), 1)  # features of user_node
        ap_feat = torch.ones(num_aps, num_aps_features)  # features of user_node
        y1 = channel_matrices[i, :, :].reshape(-1, 1)
        y2 = ap_selection_matrix[i, :, :].reshape(-1, 1)

        edge_feat_uplink = np.concatenate((y1, y2), 1)

        edge_feat_downlink = np.concatenate((y1, y2), 1)
        graph = HeteroData({
            'ue': {'x': user_feat},
            'ap': {'x': ap_feat}
        })
        # Create edge types and building the graph connectivity:
        graph['ue', 'uplink', 'ap'].edge_attr = torch.tensor(edge_feat_uplink, dtype=torch.float)
        graph['ap', 'downlink', 'ue'].edge_attr = torch.tensor(edge_feat_downlink, dtype=torch.float)

        graph['ue', 'uplink', 'ap'].edge_index = torch.tensor(adj_matrix(num_users, num_aps).transpose(),
                                                              dtype=torch.int64).contiguous()
        graph['ap', 'downlink', 'ue'].edge_index = torch.tensor(adj_matrix(num_aps, num_users).transpose(),
                                                                dtype=torch.int64).contiguous()

        # Swap
        # graph['ue', 'uplink', 'ap'].edge_index = torch.tensor(adj_matrix(num_aps, num_users).transpose(),
        #                                                       dtype=torch.int64)
        # graph['ap', 'downlink', 'ue'].edge_index = torch.tensor(adj_matrix(num_users, num_aps).transpose(),
        #                                                         dtype=torch.int64)

        # graph['ap', 'downlink', 'ue'].edge_attr  = torch.tensor(edge_feat_downlink, dtype=torch.float)
        graph_list.append(graph)
    return graph_list


def adj_matrix(num_from, num_dest):
    adj = []
    for i in range(num_from):
        for j in range(num_dest):
            adj.append([i, j])
    return np.array(adj)


# region Training and Testing functions
def loss_function(output, batch, noise_matrix, size, is_train=True, is_log=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_ue, num_ap, batch_size = size

    output = torch.reshape(output, (batch_size, num_ue, -1))
    ##
    channel_matrix = batch['ue', 'ap']['edge_attr'][:,0]
    ##
    power_max = output[:, :, 0]
    power = output[:, :, 1] * power_max
    ap_selection = batch['ue', 'ap']['edge_attr'][:, 1]
    # power_max = batch['ue']['x'][:, 0]
    # Get ap_selection from the edge_attr
    # power = batch['ue']['x'][:, 1]
    # ap_selection = batch['ue']['x'][:, 2]
    ##
    P = torch.reshape(ap_selection, (-1, num_ap, num_ue))

    G = torch.reshape(channel_matrix, (-1, num_ap, num_ue))
    # P = torch.reshape(power, (-1, num_ap, num_user)) #* p_max
    # P = torch.zeros_like(G, requires_grad=True).clone()
    # P[torch.arange(batch_size).unsqueeze(1), ap_selection, torch.arange(num_ue)] = power
    power = power.unsqueeze(1)
    P = P * power
    ##
    # new_noise = torch.from_numpy(noise_matrix).to(device)
    new_noise = noise_matrix
    desired_signal = torch.sum(torch.mul(P, G), dim=1).unsqueeze(-1)
    G_UE = torch.sum(G, dim=2).unsqueeze(-1)
    all_signal = torch.matmul(P.permute((0, 2, 1)), G_UE)
    interference = all_signal - desired_signal + new_noise
    rate = torch.log(1 + torch.div(desired_signal, interference))
    sum_rate = torch.mean(torch.sum(rate, 1))
    mean_power = torch.mean(torch.sum(P.permute((0, 2, 1)), 1))
    if is_log:
        print(f'power: {P[0]}')
        # print(f'channel: {G[0]}')
        # print(f'desired_signal: {desired_signal[0]}')
        # print(f'interference: {interference[0]}')

    if is_train:
        return torch.neg(sum_rate)  # / mean_power)
    else:
        return sum_rate  # / mean_power


def train(data_loader, noise):
    model.train()
    device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_examples = total_loss = 0
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
        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        out = out['ue']
        tmp_loss = loss_function(out, batch, noise, (num_ues, num_aps, batch_size), True)
        tmp_loss.backward()
        optimizer.step()
        total_examples += batch_size
        total_loss += float(tmp_loss) * batch_size

    return total_loss / total_examples


def test(data_loader, noise, is_log=False):
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
        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        out = out['ue']
        tmp_loss = loss_function(out, batch, noise, (num_ues, num_aps, batch_size), False)
        total_examples += batch_size
        total_loss += float(tmp_loss) * batch_size
    if is_log:
        print(out)
        # tmp_loss = loss_function(out, batch, noise, (num_ues, num_aps, batch_size), False, True)
    return total_loss / total_examples


# endregion


if __name__ == '__main__':
    K = 4  # number of APs
    N = 5  # number of nodes
    R = 10  # radius

    num_users_features = 2
    num_aps_features = 2

    num_train = 20  # number of training samples
    num_test = 4  # number of test samples

    reg = 1e-2
    pmax = 1
    var_db = 10
    var = 1 / 10 ** (var_db / 10)
    var_noise = 10e-13

    power_threshold = 2.0

    X_train, noise_train, pos_train, adj_train, index_train = generate_channels(K, N, num_train, var_noise, R)
    X_test, noise_test, pos_test, adj_test, index_test = generate_channels(K + 1, N + 10, num_test, var_noise, R)

    # theta_train = np.random.randint(K, size=(N, 1))
    # theta_test = np.random.randint(K + 1, size=(N + 10, 1))
    theta_train = np.random.randint(2, size=(num_train, K, N))
    theta_test = np.random.randint(2, size=(num_test, K+1, N+10))
    # Maybe need normalization here
    train_data = convert_to_hetero_data(X_train, power_threshold, theta_train)
    test_data = convert_to_hetero_data(X_test, power_threshold, theta_test)

    batchSize = 2

    train_loader = DataLoader(train_data, batchSize, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data, batchSize, shuffle=True, num_workers=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    data = train_data[0]
    data = data.to(device)
    #
    # # model = HGTGNN(data, hidden_channels=64, out_channels=4, num_heads=2, num_layers=1)
    # # model = model.to(device)
    #
    model = RGCN(data, num_layers=1)  # input data for the metadata (list of node types and edge types
    model = model.to(device)
    #
    # # print(data.edge_index_dict)
    # with torch.no_grad():
    #     output = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
    # print(output)

    #
    # # Training and testing
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    for epoch in range(1, 50):
        loss = train(train_loader, noise_train)
        test_acc = test(test_loader, noise_train)
        if (epoch % 5 == 1):
            test(train_loader, noise_train, True)
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Test Reward: {test_acc:.4f}')