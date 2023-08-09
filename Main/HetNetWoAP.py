import torch
import numpy as np

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader


from WSN_GNN import generate_channels_wsn
from hgt_conv import HGTGNN
from het_net_gnn import RGCN


#region Create HeteroData from the wireless system
def convert_to_hetero_data(channel_matrices):
    graph_list = []
    num_sam, num_aps, num_users = channel_matrices.shape
    for i in range(num_sam):
        x1 = torch.ones(num_users, 1)
        x2 = torch.zeros(num_users, 1)  # power allocation
        x3 = torch.ones((num_users, 1), dtype=torch.int32)  # ap selection?
        user_feat = torch.cat((x1,x2,x3),1)  # features of user_node
        ap_feat = torch.zeros(num_aps, num_aps_features)  # features of user_node
        edge_feat_uplink = channel_matrices[i, :, :].reshape(-1, 1)
        edge_feat_downlink = channel_matrices[i, :, :].reshape(-1, 1)
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


#region Training and Testing functions
def loss_function(output, batch, size, is_train=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_ue, num_ap, batch_size = size

    output = torch.reshape(output, (batch_size, num_ue, -1))
    ##
    channel_matrix = batch['ue', 'ap']['edge_attr']
    ##
    power_max = output[:, :,0]
    power = output[:, :, 1] * power_max
    ap_selection = output[:, :, 2] * num_ap
    # power_max = batch['ue']['x'][:, 0]
    # power = batch['ue']['x'][:, 1]
    # ap_selection = batch['ue']['x'][:, 2]
    ##
    ap_selection = ap_selection.int()

    G = torch.reshape(channel_matrix, (-1, num_ap, num_ue))
    # P = torch.reshape(power, (-1, num_ap, num_user)) #* p_max
    P = torch.zeros_like(G, requires_grad=True).clone()
    P[torch.arange(batch_size).unsqueeze(1), ap_selection, torch.arange(num_ue)] = power

    ##
    # new_noise = torch.from_numpy(noise_matrix).to(device)
    desired_signal = torch.sum(torch.mul(P, G), dim=1).unsqueeze(-1)
    G_UE = torch.sum(G, dim=2).unsqueeze(-1)
    all_signal = torch.matmul(P.permute((0,2,1)), G_UE)
    interference = all_signal - desired_signal  # + new_noise
    rate = torch.log(1 + torch.div(desired_signal, interference))
    sum_rate = torch.mean(torch.sum(rate, 1))
    mean_power = torch.mean(torch.sum(P.permute((0,2,1)), 1))

    if is_train:
        return torch.neg(sum_rate )#/ mean_power)
    else:
        return sum_rate #/ mean_power


def train(data_loader):
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
        tmp_loss = loss_function(out, batch, (num_ues, num_aps, batch_size), True)
        tmp_loss.backward()
        optimizer.step()
        #total_examples += batch_size
        total_loss += float(tmp_loss) #* batch_size

    return total_loss #/ total_examples


def test(data_loader):
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
        tmp_loss = loss_function(out, batch, (num_ues, num_aps, batch_size), False)
        #total_examples += batch_size
        total_loss += float(tmp_loss) #* batch_size

    return total_loss #/ total_examples
#endregion


if __name__ == '__main__':
    K = 4  # number of APs
    N = 5  # number of nodes
    R = 10  # radius

    num_users_features = 3
    num_aps_features = 3

    num_train = 2  # number of training samples
    num_test = 4  # number of test samples

    reg = 1e-2
    pmax = 1
    var_db = 10
    var = 1 / 10 ** (var_db / 10)
    var_noise = 10e-11

    power_threshold = 2.0

    X_train, noise_train, pos_train, adj_train, index_train = generate_channels_wsn(K, N, num_train, var_noise, R)
    X_test, noise_test, pos_test, adj_test, index_test = generate_channels_wsn(K + 1, N + 10, num_test, var_noise, R)

    # Maybe need normalization here
    train_data = convert_to_hetero_data(X_train)
    test_data = convert_to_hetero_data(X_test)

    batchSize = 2

    train_loader = DataLoader(train_data, batchSize, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data, batchSize, shuffle=True, num_workers=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = train_data[0]
    data = data.to(device)

    # model = HGTGNN(data, hidden_channels=64, out_channels=4, num_heads=2, num_layers=1)
    # model = model.to(device)

    model = RGCN(data, num_layers=1)  # input data for the metadata (list of node types and edge types
    model = model.to(device)

    # # # print(data.edge_index_dict)
    # with torch.no_grad():
    #     output = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
    # print(output)


    # Training and testing
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    for epoch in range(1, 20):
        loss = train(train_loader)
        if epoch % 8 == 1:
            test_acc = test(test_loader)
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Test Reward: {test_acc:.4f}')

