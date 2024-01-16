import torch
import numpy as np
from torch_geometric.data import HeteroData


#region Create HeteroData from the wireless system
def convert_to_hetero_data(channel_matrices, p_max, ap_selection_matrix):
    graph_list = []
    num_sam, num_aps, num_users = channel_matrices.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(num_sam):
        x1 = torch.ones(num_users, 1) * p_max
        x2 = torch.ones(num_users, 1)  # power allocation
        ap_feat = torch.zeros(num_aps, 0)  # features of ap_node
        y1 = channel_matrices[i, :, :].reshape(-1, 1)
        y2 = ap_selection_matrix[i, :, :].reshape(-1, 1)
        edge_feat_downlink = np.concatenate((y1, y2), 1)

        y1 = channel_matrices[i, :, :].T.reshape(-1, 1)
        y2 = ap_selection_matrix[i, :, :].T.reshape(-1, 1)
        edge_feat_uplink = np.concatenate((y1, y2), 1)
        graph = HeteroData({
            'ue': {'x': x1.to(device)},
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