import time
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid
from torch_geometric.loader import DataLoader

from Main.Utilities.setup import get_arguments
# from Main.Utilities.load_file import load_data_from_mat
from Main.het_net_gnn import Ue2Ap, PowerConv_wAP, ApSelectConv
from Main.HetNet_AP import data_prepare, convert_to_hetero_data


def mlp(channels):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i], bias=True), ReLU())
        for i in range(1, len(channels))
    ])


class HetNetGNN_v4(nn.Module):
    def __init__(self, dataset):
        super(HetNetGNN_v4, self).__init__()

        out_dim = 32
        self.convs = torch.nn.ModuleList()
        self.conv1 = Ue2Ap(node_dim={'ue': 1, 'ap': 0}, edge_dim=2,
                         out_node_dim = out_dim, metadata=dataset.metadata(), aggr='add')
        self.conv2 = ApSelectConv(node_dim={'ue': out_dim, 'ap': out_dim}, edge_dim=2,
                         out_node_dim = out_dim, metadata=dataset.metadata(), aggr='add')
        self.conv3 = PowerConv_wAP(node_dim={'ue': out_dim, 'ap': out_dim}, edge_dim=2,
                         out_node_dim = out_dim, metadata=dataset.metadata(), aggr='add')
        self.conv4 = PowerConv_wAP(node_dim={'ue': out_dim, 'ap': out_dim}, edge_dim=2,
                         out_node_dim = out_dim, metadata=dataset.metadata(), aggr='add')

        self.power_mlp = mlp([out_dim, 16])
        self.power_mlp = Seq(*[self.power_mlp, Seq(Lin(16, 1, bias=True), Sigmoid())])

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict, edge_attr_dict = self.conv1(x_dict=x_dict, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict)
        # a = x_dict['ue']
        # b = x_dict['ap']
        # print(f'=== Layer 1: ue: {a.shape}, ap: {a.shape}')
        x_dict, edge_attr_dict = self.conv2(x_dict=x_dict, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict)
        x_dict, edge_attr_dict = self.conv3(x_dict=x_dict, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict)
        x_dict, edge_attr_dict = self.conv4(x_dict=x_dict, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict)
        ue_feat = x_dict['ue']
        # print(ue_feat)
        power = self.power_mlp(ue_feat)

        x_dict['ue'] = torch.cat([x_dict['ue'][:,:1], power], dim=1)
        # print(x_dict['ue'])

        return x_dict, edge_attr_dict


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


def loss_function(power_out, edge_dict, noise_matrix, size, p_cir, loss_type, is_train=True):
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

    power_consumed = power_all + p_cir
    sum_rate_batch = torch.sum(rate, dim=1)
    sum_power_batch = torch.sum(power_consumed, dim=1)

    if loss_type == 'GlobalEE':
        # Global Energy Efficiency = sum(Rate)/sum(Power)
        ee_batch = torch.div(sum_rate_batch, sum_power_batch)
    elif loss_type == 'SumEE':
        # Sum Energy Efficiency = sum(Rate/Power)
        ee_batch = torch.sum(torch.div(rate, power_consumed), dim=1)
    elif loss_type == 'ProdEE':
        # Product Energy Efficiency = Product(Rate/Power)
        ee_batch = torch.prod(torch.div(rate, power_consumed), dim=1)
    else:
        raise RuntimeError("Loss Function not Defined!")
    ee_mean = torch.mean(ee_batch)

    if is_train:
        return sum_rate, torch.neg(ee_mean), torch.mean(sum_power_batch)

    else:
        return torch.neg(ee_mean)


def train(data_loader, noise, p_cir, model, loss_type, optimizer):
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
        tmp_sumRate, tmp_loss, tmp_sumPower = loss_function(out, edge, noise, (num_ues, num_aps, batch_size), p_cir,
                                                            loss_type, True)
        tmp_loss.backward()
        optimizer.step()
        total_examples += batch_size
        total_loss += float(tmp_loss) * batch_size
        sumRate += float(tmp_sumRate) * batch_size
        sumPower += float(tmp_sumPower) * batch_size

    return sumRate / total_examples, total_loss / total_examples, sumPower / total_examples


def test(data_loader, noise, p_cir, model, loss_type):
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

        tmp_loss = loss_function(out, edge, noise, (num_ues, num_aps, batch_size), p_cir, loss_type, False)
        total_examples += batch_size
        total_loss += float(tmp_loss) * batch_size

    last_batch_edge = (edge['ue','uplink','ap'])

    return total_loss / total_examples, last_batch_edge


def main(args):
    # Get arguments
    args = get_arguments()

    power_threshold = args.poweru_max
    power_circuit = args.power_cir

    X_train, theta_train, noise_train, theta_train_dummy, X_test, \
        theta_test, noise_test, theta_test_dummy = data_prepare(args)

    train_data = convert_to_hetero_data(X_train, power_threshold, theta_train_dummy)
    test_data = convert_to_hetero_data(X_test, power_threshold, theta_test_dummy)

    batchSize = args.batch_size

    train_loader = DataLoader(train_data, batchSize, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batchSize, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = train_data[0]
    data = data.to(device)

    model = HetNetGNN_v4(data)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    training_loss = []
    testing_acc = []

    for epoch in range(1, args.epoch_num):
        train_sumrate, loss, train_sumPower = train(train_loader, noise_train, power_circuit, model, args.loss_type, optimizer)
        test_acc, last_batch_edge1 = test(test_loader, noise_test, power_circuit, model, args.loss_type)
        training_loss.append(loss)
        testing_acc.append(test_acc)
        scheduler.step()
        if (epoch % args.per_epoch == 1):
            # tmp = test(test_loader, noise_test, True)
            # sumrate.append(float(tmp))
            print(
                f'Epoch: {epoch:03d}, Train Loss: {loss:.6f}, Train Sum Rate: {train_sumrate:.4f}, Train Sum Power: {train_sumPower:.0f}, Test Reward: {test_acc:.6f}')


    return training_loss, testing_acc


if __name__ == "__main__":
    total_start = time.time()

    arguments = get_arguments()

    train_loss, test_lost = main(arguments)

    # print(training_loss)


