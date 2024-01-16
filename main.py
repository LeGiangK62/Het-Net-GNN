import time
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid
from torch_geometric.loader import DataLoader

from Main.Utilities.load_file import loading_data, load_data_from_mat
from Main.Utilities.setup import get_arguments
from Main.het_net_gnn import Ue2Ap, EdgeConv, PowerConv, mlp
from Main.Utilities.data_load import convert_to_hetero_data
from Main.Utilities.train_test_function import train, test


class HetNetGNN(nn.Module):
    def __init__(self, dataset, debugMode=False):
        super(HetNetGNN, self).__init__()
        self.debugMode = debugMode
        out_dim = 32
        edge_out_dim = 8
        self.convs = torch.nn.ModuleList()
        self.conv1 = Ue2Ap(node_dim={'ue': 1, 'ap': 0},
                           edge_dims={'in': 2, 'out': edge_out_dim},
                           out_node_dim=out_dim, metadata=dataset.metadata(), aggr='add')
        self.conv2 = EdgeConv(node_dim={'ue': out_dim, 'ap': out_dim},
                              edge_dim=edge_out_dim,
                              out_node_dim=out_dim, metadata=dataset.metadata(), aggr='add')
        self.conv3 = PowerConv(node_dim={'ue': out_dim, 'ap': out_dim}, edge_dim=edge_out_dim,
                               out_node_dim=out_dim, metadata=dataset.metadata(), aggr='add')

        self.power_mlp = mlp([out_dim, 16])
        self.power_mlp = Seq(*[self.power_mlp, Seq(Lin(16, 1, bias=True), Sigmoid())])

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict, edge_attr_dict = self.conv1(x_dict=x_dict, edge_index_dict=edge_index_dict,
                                            edge_attr_dict=edge_attr_dict)
        x_dict, edge_attr_dict = self.conv2(x_dict=x_dict, edge_index_dict=edge_index_dict,
                                            edge_attr_dict=edge_attr_dict)
        x_dict, edge_attr_dict = self.conv3(x_dict=x_dict, edge_index_dict=edge_index_dict,
                                            edge_attr_dict=edge_attr_dict)

        ue_feat = x_dict['ue']
        power = self.power_mlp(ue_feat)

        x_dict['ue'] = torch.cat([x_dict['ue'][:, :1], power], dim=1)
        return x_dict, edge_attr_dict


class HetNetGNN_noAP(nn.Module):
    def __init__(self, dataset):
        super(HetNetGNN_noAP, self).__init__()

        out_dim = 32
        edge_out_dim = 8
        self.convs = torch.nn.ModuleList()
        self.conv1 = Ue2Ap(node_dim={'ue': 1, 'ap': 0},
                           edge_dims={'in': 2, 'out': edge_out_dim},
                           out_node_dim=out_dim, metadata=dataset.metadata(), aggr='add')
        self.conv3 = PowerConv(node_dim={'ue': out_dim, 'ap': out_dim}, edge_dim=edge_out_dim,
                               out_node_dim=out_dim, metadata=dataset.metadata(), aggr='add')
        self.power_mlp = mlp([out_dim, 16])
        self.power_mlp = Seq(*[self.power_mlp, Seq(Lin(16, 1, bias=True), Sigmoid())])

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict, edge_attr_dict = self.conv1(x_dict=x_dict, edge_index_dict=edge_index_dict,
                                            edge_attr_dict=edge_attr_dict)
        x_dict, edge_attr_dict = self.conv3(x_dict=x_dict, edge_index_dict=edge_index_dict,
                                            edge_attr_dict=edge_attr_dict)
        ue_feat = x_dict['ue']
        power = self.power_mlp(ue_feat)

        x_dict['ue'] = torch.cat([x_dict['ue'][:, :1], power], dim=1)

        return x_dict, edge_attr_dict


def main_run(net_type, metadata, train_sample, noise_train, test_sample, noise_test, power_circuit,
             num_epoch=600, valid_epoch=100, lr=1e-3, debug_mode=False,
             eps=1e-3, reg=0.01):
    if net_type not in ["wAP", "woAP", 'combined']:
        raise ValueError("Invalid input type. Please use 'wAP' or 'woAP'.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = metadata[0]
    data = data.to(device)
    if net_type == "woAP":
        model = HetNetGNN_noAP(data)
    else:
        model = HetNetGNN(data, debug_mode)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    training_loss = []
    testing_reward= []
    training_reward = []
    flag = False
    for epoch in range(1, num_epoch+1):
        train_sumrate, loss, train_sumPower, train_ee = train(model, optimizer, train_sample,
                                                              noise_train, power_circuit, eps, reg)
        test_acc = test(model, test_sample, noise_test, power_circuit)
        if loss == 0.0 or test_acc == 0.0:
            flag = True
            break
        training_loss.append(loss)
        testing_reward.append(test_acc)
        training_reward.append(train_ee)
        scheduler.step()
        if epoch % valid_epoch == 1:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.6f}, Train Energy Efficiency: {train_ee:.4f}, Train Sum '
                  f'Rate: {train_sumrate:.4f}, Train Sum Power: {train_sumPower:.0f}, Test Reward: {test_acc:.6f}')

    if flag:
        print('Re-running...')
        return main_run(net_type, metadata, train_sample, noise_train, test_sample, noise_test, power_circuit,
                        num_epoch, valid_epoch, lr)

    return model, training_loss, testing_reward, training_reward


def main(args):
    power_threshold = args.poweru_max
    power_circuit = args.power_cir

    batchSize = args.batch_size
    num_train = args.train_num
    num_test = args.test_num

    K = args.ap_num  # number of APs
    N = args.user_num  # number of nodes
    R = args.radius  # radius

    K_test = K
    N_test = N

    mat_file = args.mat_file
    ##############

    channel_load, theta_load, power, EE_load, bandW, noise, (num_s, num_aps, num_ues) = load_data_from_mat(mat_file, 
                                                                                                           args.default_folder)
    shuffled_indices = np.arange(num_s)
    np.random.shuffle(shuffled_indices)

    channel_load = channel_load[shuffled_indices]
    theta_load = theta_load[shuffled_indices]
    power = power[shuffled_indices]

    (X_train, theta_train, noise_train, EE_res_train), (X_test, theta_test, noise_test, EE_res_test) = loading_data(
        num_train, num_test, channel_load, theta_load, power, EE_load, noise)

    theta_train = np.zeros((num_train, K, N))
    theta_test = np.zeros((num_test, K_test, N_test))

    for sample_idx in range(theta_train.shape[0]):
        for col_idx in range(theta_train.shape[2]):
            row_idx = np.random.choice(theta_train.shape[1])
            theta_train[sample_idx, row_idx, col_idx] = 1

    for sample_idx in range(theta_test.shape[0]):
        for col_idx in range(theta_test.shape[2]):
            row_idx = np.random.choice(theta_test.shape[1])
            theta_test[sample_idx, row_idx, col_idx] = 1

    train_data = convert_to_hetero_data(X_train, power_threshold, theta_train)
    test_data = convert_to_hetero_data(X_test, power_threshold, theta_test)

    train_loader = DataLoader(train_data, batchSize, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batchSize, shuffle=True, num_workers=0)

    model, training_loss, testing_acc, training_acc = main_run(args.model_mode,
                                                               metadata=train_data, train_sample=train_loader,
                                                               noise_train=noise_train,
                                                               test_sample=test_loader, noise_test=noise_test,
                                                               power_circuit=power_circuit,
                                                               num_epoch=args.epoch_num, valid_epoch=args.valid_epoch,
                                                               lr=args.lr, debug_mode=False,
                                                               eps=args.eps, reg=args.reg
                                                               )
    return training_loss, testing_acc


if __name__ == "__main__":
    total_start = time.time()

    arguments = get_arguments()

    train_loss, test_lost = main(arguments)

    # print(training_loss)


