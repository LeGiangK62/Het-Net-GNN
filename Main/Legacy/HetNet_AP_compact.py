import torch
import numpy as np
import h5py
from scipy.spatial import distance_matrix

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

# from WSN_GNN import generate_channels_wsn
# from hgt_conv import HGTGNN
from .GNN_AP import RGCN
from Main.Utilities.setup import get_arguments


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


# region Training and Testing functions
# Unsupervised learning
def loss_function(output, batch, noise_matrix, size, p_cir, is_train=True, is_log=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_ue, num_ap, batch_size = size

    output = torch.reshape(output, (batch_size, num_ue, -1))
    channel_matrix = batch['ue', 'ap']['edge_attr'][:, 0]
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

def loss_function_fixed(power_out, edge_dict, noise_matrix, size, p_cir, is_train=True, is_log=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_ue, num_ap, batch_size = size

    power_out = torch.reshape(power_out, (batch_size, num_ue, -1))
    channel_matrix = edge_dict['ue', 'uplink', 'ap'][:, 0]
    power_max = power_out[:, :, 0]
    power = power_out[:, :, 1] * power_max
    ap_selection = edge_dict['ue', 'uplink', 'ap'][:, 1]
    P = torch.reshape(ap_selection, (-1, num_ap, num_ue))

    G = torch.reshape(channel_matrix, (-1, num_ap, num_ue))
    power = power.unsqueeze(1)
    # P = P * power
    sum_rate, rate = sum_rate_calculation(P * power, P, G, noise_matrix)
    power_all = torch.sum(power, 1).unsqueeze(-1)

    # ee = torch.mean( torch.div(rate, power_all + p_cir)) # Personal Energy efficiency
    ee = torch.mean(torch.div(rate, power_all + p_cir))  # Option 1
    sum_rate_batch = torch.sum(rate, dim=1)
    sum_power_batch = torch.sum(power_all + p_cir, dim=1)
    ee_batch = torch.div(sum_rate_batch, sum_power_batch)
    ee_mean = torch.mean(ee_batch)

    if is_log:
        print(f'power: {P[0]}')
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
    channel_matrix = batch['ue', 'ap']['edge_attr'][:, 0]
    power_max = output[:, :, 0]
    power = output[:, :, 1] * power_max
    ap_selection = batch['ue', 'ap']['edge_attr'][:, 1]
    P = torch.reshape(ap_selection, (-1, num_ap, num_ue))

    G = torch.reshape(channel_matrix, (-1, num_ap, num_ue))
    power = power.unsqueeze(1)
    P = P * power
    sum_rate, _ = sum_rate_calculation(P, G, noise_matrix)
    return sum_rate


def train(model, optimizer, data_loader, noise, p_cir):
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
        tmp_sumRate, tmp_loss, tmp_sumPower = loss_function(out, batch, noise, (num_ues, num_aps, batch_size),
                                                            p_cir, True)
        tmp_loss.backward()
        optimizer.step()
        total_examples += batch_size
        total_loss += float(tmp_loss) * batch_size
        sumRate += float(tmp_sumRate) * batch_size
        sumPower += float(tmp_sumPower) * batch_size

    return sumRate / total_examples, total_loss / total_examples, sumPower / total_examples


def test(model, data_loader, noise, p_cir, is_log=False):
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

    last_batch_edge = (edge['ue', 'uplink', 'ap'])
    if is_log:
        # print(out[:3])
        tensor = (edge['ue', 'uplink', 'ap'])
        # print(tensor)
        print(f'The number of activated link: {(tensor[:, 1] == 0).sum().item()}/{len(tensor[:, 1])}')
        # tmp_loss = loss_function(out, batch, noise, (num_ues, num_aps, batch_size), False, True)
    return total_loss / total_examples, last_batch_edge


def sum_rate_calculation(power_matrix, ap_selection, channel_matrix, noise_matrix):
    P = power_matrix
    G = channel_matrix
    new_noise = noise_matrix
    desired_signal = torch.sum(torch.mul(P, G), dim=1).unsqueeze(-1)
    P_trans = P.permute(0, 2, 1)
    P_UE = torch.sum(P_trans, dim=2).unsqueeze(-1)  # P_UE[n] = The power n-th UE transmits
    all_received_signal = torch.matmul(G, P_UE)
    all_signal = torch.matmul(ap_selection.permute(0, 2, 1), all_received_signal)
    # max_P,_ = torch.max(P_trans, dim=2)
    # max_P = max_P.unsqueeze(-1)
    # print(max_P)
    # all_signal = torch.div(tmp, max_P)
    interference = -desired_signal + all_signal + noise_matrix
    rate = torch.log(1 + torch.div(desired_signal, interference))
    sum_rate = torch.mean(torch.sum(rate, 1))
    return sum_rate, rate
# endregion


# Supervised learning
def load_data_from_mat(file_path):
    loaded_data = {}
    with h5py.File(file_path, 'r') as file:
        for key in file.keys():
            loaded_data[key] = file[key][()]
    return loaded_data

def generate_data_loaders(num_train, num_test, num_ap, num_ue, noise, radius, p_max, gen_data_flag=True):
    if gen_data_flag:
        channel_train, noise_var_train, pos_train, adj_train, index_train = generate_channels(num_ap, num_ue, num_train,
                                                                                              noise, radius)
        channel_test, noise_var_test, pos_test, adj_test, index_test = generate_channels(num_ap, num_ue, num_test,
                                                                                         noise, radius)

        theta_train = np.zeros((num_train, num_ap, num_ue))
        theta_test = np.zeros((num_test, num_ap, num_ue))
        # np.random.randint(2, size=(num_train, K, N))
        # theta_test = np.random.randint(2, size=(num_test, K_test, N_test))

        for sample_idx in range(theta_train.shape[0]):
            for col_idx in range(theta_train.shape[2]):
                row_idx = np.random.choice(theta_train.shape[1])
                theta_train[sample_idx, row_idx, col_idx] = 1

        for sample_idx in range(theta_test.shape[0]):
            for col_idx in range(theta_test.shape[2]):
                row_idx = np.random.choice(theta_test.shape[1])
                theta_test[sample_idx, row_idx, col_idx] = 1
    else:
        mat_file = '../../Data/no_time_allo_train_20Aug.mat'

        channel_load, theta_load, power, EE_result, bandW, noise, (num_s, num_aps, num_ues) = load_data_from_mat(
            mat_file)

        shuffled_indices = np.arange(num_s)
        np.random.shuffle(shuffled_indices)

        channel_load = channel_load[shuffled_indices]
        theta_load = theta_load[shuffled_indices]
        power = power[shuffled_indices]

        channel_train, ap_select_train, noise_var_train = channel_load[0:num_train] ** 2 / noise, \
            theta_load[0:num_train], 1
        channel_test, ap_selection_test, noise_var_test = channel_load[num_train:(num_train + num_test)] ** 2 / noise, \
            theta_load[num_train:(num_train + num_test)], 1

    train_data_load = convert_to_hetero_data(channel_train, p_max, theta_train)
    test_data_load = convert_to_hetero_data(channel_test, p_max, theta_test)

    return train_data_load, test_data_load, noise_var_train, noise_var_test


def main_train(args):
    # Get arguments
    args = get_arguments()

    K = args.ap_num  # number of APs
    N = args.user_num  # number of nodes
    R = args.radius  # radius

    isGenData = args.generate_data

    n0 = args.noise

    power_threshold = args.poweru_max
    power_circuit = args.power_cir

    K_test = K
    N_test = N

    networkMode = args.model_mode

    if networkMode != 'withAP':
        raise RuntimeError("Not yet Define this mode")

    num_train = args.train_num  # number of training samples
    num_test = args.test_num  # number of test samples

    train_data, test_data, noise_train, noise_test = generate_data_loaders(num_train, num_test, K, N,
                                                                           n0, R, power_threshold, isGenData)

    batchSize = args.batch_size

    train_loader = DataLoader(train_data, batchSize, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data, batchSize, shuffle=True, num_workers=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = train_data[0]
    data = data.to(device)
    model = RGCN(data, num_layers=3)  # input data for the metadata (list of node types and edge types)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    training_loss = []
    testing_acc = []
    sumrate = []
    change_count = 0
    for epoch in range(1, args.epoch_num):
        train_sumrate, loss, train_sumPower = train(model, optimizer, train_loader, noise_train, power_circuit)
        test_acc, last_batch_edge1 = test(model, test_loader, noise_test, power_circuit)
        training_loss.append(loss)
        testing_acc.append(test_acc)
        sumrate.append(float(train_sumrate))
        ##
        tmp, last_batch_edge2 = test(model, test_loader, noise_test, power_circuit)
        e1 = last_batch_edge1[:, 1]
        e2 = last_batch_edge2[:, 1]
        if torch.sum(e1 * e2) != torch.sum(e1):
            # print("Link Changed")
            change_count = change_count + 1

        if (epoch % args.per_epoch == 1):
            # tmp = test(test_loader, noise_test, True)
            # sumrate.append(float(tmp))
            print(
                f'Epoch: {epoch:03d}, Train Loss: {loss:.6f}, Train Sum Rate: {train_sumrate:.4f}, Train Sum Power: {train_sumPower:.0f}, Test Reward: {test_acc:.6f}')

    return training_loss, testing_acc
#
# if __name__ == '__main__':
#     K = 2  # number of APs
#     N = 3  # number of nodes
#     R = 1000  # radius
#     K_test = K
#     N_test = N
#
#     num_users_features = 2
#     num_aps_features = 2
#
#     num_train = 1  # number of training samples
#     num_test = 4  # number of test samples
#
#     reg = 1e-2
#     pmax = 1
#     var_db = 10
#     var = 1 / 10 ** (var_db / 10)
#     var_noise = 10e-12
#     # var_noise = 1
#
#     power_threshold = 200
#     power_circuit = 1.0
#
#     # X_train, noise_train, pos_train, adj_train, index_train = generate_channels(K, N, num_train, var_noise, R)
#     # X_test, noise_test, pos_test, adj_test, index_test = generate_channels(K_test, N_test, num_test, var_noise, R)
#     #
#     # theta_train = np.zeros((num_train, K, N))
#     # theta_test = np.zeros((num_test, K_test, N_test))
#     # # np.random.randint(2, size=(num_train, K, N))
#     # # theta_test = np.random.randint(2, size=(num_test, K_test, N_test))
#     #
#     # for sample_idx in range(theta_train.shape[0]):
#     #     for col_idx in range(theta_train.shape[2]):
#     #         row_idx = np.random.choice(theta_train.shape[1])
#     #         theta_train[sample_idx, row_idx, col_idx] = 1
#     #
#     # for sample_idx in range(theta_test.shape[0]):
#     #     for col_idx in range(theta_test.shape[2]):
#     #         row_idx = np.random.choice(theta_test.shape[1])
#     #         theta_test[sample_idx, row_idx, col_idx] = 1
#     # # Maybe need normalization here
#     # train_data = convert_to_hetero_data(X_train, power_threshold, theta_train)
#     # test_data = convert_to_hetero_data(X_test, power_threshold, theta_test)
#
#     train_data, test_data, noise_train, noise_test = generate_data_loaders(power_threshold, power_circuit, False)
#
#     batchSize = 2
#
#     train_loader = DataLoader(train_data, batchSize, shuffle=True, num_workers=1)
#     test_loader = DataLoader(test_data, batchSize, shuffle=True, num_workers=1)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     data = train_data[0]
#     data = data.to(device)
#     # # model = HGTGNN(data, hidden_channels=64, out_channels=4, num_heads=2, num_layers=1)
#     # # model = model.to(device)
#     #
#     model = RGCN(data, num_layers=1)  # input data for the metadata (list of node types and edge types
#     model = model.to(device)
#     #
#     # # # print(data.edge_index_dict)
#     # # with torch.no_grad():
#     # #     output = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
#     # # print(output)
#     #
#     # #
#     # # Training and testing
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
#     training_loss = []
#     testing_acc = []
#     sumrate = []
#     for epoch in range(1, 1000):
#         train_sumrate, loss, train_sumPower = train(train_loader, noise_train, power_circuit)
#         test_acc = test(test_loader, noise_test, power_circuit)
#         training_loss.append(loss)
#         testing_acc.append(test_acc)
#         sumrate.append(float(train_sumrate))
#         if (epoch % 100 == 1):
#             # tmp = test(test_loader, noise_test, True)
#             # tmp = test(test_loader, noise_test, power_circuit, True)
#             # sumrate.append(float(tmp))
#             print(
#                 f'Epoch: {epoch:03d}, Train Loss: {loss:.6f}, Train Sum Rate: {train_sumrate:.4f}, Train Sum Power: {train_sumPower:.0f}, Test Reward: {test_acc:.6f}')
