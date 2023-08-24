from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing, HeteroConv
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import EdgeType, Metadata, NodeType, SparseTensor
from torch_geometric.nn.inits import reset

from torch_geometric.utils import softmax
from torch_sparse import SparseTensor


def mlp(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i], bias=True), ReLU())  # , BN(channels[i]))
        for i in range(1, len(channels))
    ])


# class EdgeConv(MessagePassing):
#     def __init__(self, node_dim, edge_dim, metadata: Metadata, aggr='mean', **kwargs):
#         super(EdgeConv, self).__init__(aggr=aggr)
#
#         self.lin_node = ModuleDict()
#         self.lin_edge = ModuleDict()
#         self.res_lin = ModuleDict()
#
#         for node_type in metadata[0]:
#             self.lin_node[node_type] = mlp([node_dim, 32])
#             self.res_lin[node_type] = nn.Linear(node_dim, 32)
#
#         for edge_type in metadata[1]:
#             self.lin_edge['__'.join(edge_type)] = mlp([edge_dim, 32])
#
#         self.power_mlp = mlp([32 + 3, 16])
#         self.power_mlp = Seq(*[self.power_mlp, Seq(Lin(16, 1, bias=True), Sigmoid())])
#         self.ap_mlp = mlp([32 + 3, 16])
#         self.ap_mlp = Seq(*[self.ap_mlp, Seq(Lin(16, 1, bias=True), Sigmoid())])
#
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         super().reset_parameters()
#         reset(self.lin_node)
#         reset(self.lin_edge)
#         reset(self.res_lin)
#         reset(self.power_mlp)
#         reset(self.ap_mlp)
#
#     def forward(
#         self,
#         x_dict: Dict[NodeType, Tensor],
#         edge_index_dict: Union[Dict[EdgeType, Tensor],
#                                Dict[EdgeType, SparseTensor]],  # Support both.
#         edge_attr_dict: Union[Dict[EdgeType, Tensor],
#                               Dict[EdgeType, SparseTensor]]
#     ) -> Dict[NodeType, Optional[Tensor]]:
#         # How to get the edge attributes from only the index?
#         lin_node_dict, lin_edge_dict, out_dict = {}, {}, {}
#         # Iterate over node-types to initialize the output dictionary
#         for node_type, node_ft in x_dict.items():
#             lin_node_dict[node_type] = self.lin_node[node_type](node_ft)
#             out_dict[node_type] = []
#
#         # Iterate over edge-types:
#         for edge_type, edge_index in edge_index_dict.items():
#             # aggregate information to the destination node
#             # for each type of edge
#             edge_attr = edge_attr_dict[edge_type]
#             src_type, _, dst_type = edge_type
#             edge_type = '__'.join(edge_type)
#             x_j = x_dict[src_type]
#             x_i = x_dict[dst_type]
#
#             out = self.propagate(edge_index, x=x_j, node_feat=x_i, src_type=src_type, edge_type=edge_type,
#                                  dst_type=dst_type, edge_attr=edge_attr, size=(x_j.shape[0], x_i.shape[0]))
#             out_dict[dst_type] = out
#
#         # Iterate over node-types:
#         # for node_type, outs in out_dict.items():
#             # out = group(outs, self.group)
#             #
#             # if out is None:
#             #     out_dict[node_type] = None
#             #     continue
#             #
#             # out = self.a_lin[node_type](F.gelu(out))
#             # if out.size(-1) == x_dict[node_type].size(-1):
#             #     alpha = self.skip[node_type].sigmoid()
#             #     out = alpha * out + (1 - alpha) * x_dict[node_type]
#             # out_dict[node_type] = out
#
#         return out_dict
#
#     # def message(self, x_j: Tensor, src_type, edge_type, node_mlp, edge_rel) -> Tensor:
#     def message(self, x_j: Tensor, src_type, edge_type, edge_attr) -> Tensor:
#         # This function is called when we use self.propagate - Used the given parameters too.
#         # What each neighbor node send to target along the edges
#         # Adding the edge relation here
#         node_mlp = self.lin_node[src_type]
#         edge_mlp = self.lin_edge[edge_type]
#         return node_mlp(x_j) + edge_mlp(edge_attr)
#
#     def update(self, aggr_out, node_feat, dst_type, edge_index):
#         # Update node representations with the aggregated messages
#         # aggr_out  = output of aggregation function, the following is the input of the propagation function
#         power_max = node_feat[:, 0]
#         node_mlp = self.lin_node[dst_type]
#         res = node_mlp(node_feat)
#         tmp = torch.cat([node_feat, aggr_out + res], dim=1)
#         power = self.power_mlp(tmp)
#         ap_selection = self.ap_mlp(tmp)
#         return torch.cat([power_max.unsqueeze(-1), power, ap_selection], dim=1)
#
#
# class RGCN(nn.Module):
#     def __init__(self, dataset, num_layers):
#         # The only things need to fix here are the dimensions
#         super().__init__()
#         self.convs = torch.nn.ModuleList()
#         self.mlp = mlp([32, 16])
#         for _ in range(num_layers):
#             conv = EdgeConv(node_dim=3, edge_dim=1,
#                             metadata=dataset.metadata())
#             self.convs.append(conv)
#
#
#     def forward(self, x_dict, edge_index_dict, edge_attr_dict):
#         for conv in self.convs:
#             x_dict = conv(x_dict=x_dict, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict)
#             # x_dict = {key: x.relu() for key, x in x_dict.items()}
#         return x_dict


class Ue2Ap(MessagePassing):
    def __init__(self, node_dim, edge_dim, metadata: Metadata, aggr='mean', **kwargs):
        super(Ue2Ap, self).__init__(aggr=aggr)
        self.lin_node_msg_compact = ModuleDict()
        self.lin_edge_compact = ModuleDict()
        self.lin_node_upd = ModuleDict()
        self.power_mlp = ModuleDict()

        for node_type in metadata[0]:
            self.lin_node_msg_compact[node_type] = mlp([1, 16, 32])
            self.lin_node_upd[node_type] = mlp([node_dim[node_type], 16, 32])
            # self.lin_node_upd[node_type] = mlp([1, 16, 32])
            # self.power_mlp[node_type] = mlp([32 + node_dim[node_type], 16])
            self.power_mlp[node_type] = mlp([32, 16])
            self.power_mlp[node_type] = Seq(*[self.power_mlp[node_type], Seq(Lin(16, 1, bias=True), Sigmoid())])

        for edge_type in metadata[1]:
            self.lin_edge_compact['__'.join(edge_type)] = mlp([edge_dim - 1, 16, 32])

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.power_mlp)
        reset(self.lin_edge_compact)
        reset(self.lin_node_msg_compact)
        reset(self.lin_node_upd)

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Union[Dict[EdgeType, Tensor],
            Dict[EdgeType, SparseTensor]],
            edge_attr_dict: Union[Dict[EdgeType, Tensor],
            Dict[EdgeType, SparseTensor]]
    ) -> tuple[Dict[NodeType, Optional[Tensor]], Union[Dict[EdgeType, Tensor], Dict[EdgeType, SparseTensor]]]:
        out_node_dict = {}

        # Iterate over edge-types to update node_features:
        for edge_type, edge_index in edge_index_dict.items():
            edge_attr = edge_attr_dict[edge_type]
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            x_j = x_dict[src_type]
            x_i = x_dict[dst_type]

            out = self.propagate(edge_index, x=x_j, node_feat=x_i, src_type=src_type, edge_type=edge_type,
                                 dst_type=dst_type, edge_attr=edge_attr, size=(x_j.shape[0], x_i.shape[0]))
            out_node_dict[dst_type] = out
        return out_node_dict, edge_attr_dict

    def message(self, x_j: Tensor, src_type, edge_type, edge_attr) -> Tensor:
        # APs only send the edge_feat (only the channel) to UEs
        # Ues send the node_ft (only the p_max) and the edge_ft (only the channel)

        node_mlp = self.lin_node_msg_compact[src_type]
        edge_mlp = self.lin_edge_compact[edge_type]
        node_ft = x_j[:, :1]
        edge_ft = edge_attr[:, :1]

        ap_selection = edge_attr[:, 1].unsqueeze(-1)
        edge_ft = edge_ft * (1 - ap_selection)
        if src_type == 'ue':
            return node_mlp(node_ft) + edge_mlp(edge_ft)
        else:
            return edge_mlp(edge_ft)

    def update(self, aggr_out, node_feat, dst_type):
        power_max = node_feat[:, 0]
        node_mlp = self.lin_node_upd[dst_type]
        final_mlp = self.power_mlp[dst_type]
        res = node_mlp(node_feat)
        # tmp = torch.cat([node_feat, aggr_out + res], dim=1)
        # power = final_mlp(tmp)
        power = final_mlp(aggr_out + res)
        # return torch.cat([node_feat[:, :-1], power], dim=1)
        if dst_type == 'ue':
            return torch.cat([power_max.unsqueeze(-1), power], dim=1)
        else:
            return power


class PowerConv(MessagePassing):
    def __init__(self, node_dim: Dict, edge_dim, metadata: Metadata, ap_selection=False, aggr='mean', **kwargs):
        super(PowerConv, self).__init__(aggr=aggr)
        self.is_ap_selection = ap_selection
        self.lin_node_msg = ModuleDict()
        self.lin_edge_compact = ModuleDict()
        self.lin_node_upd = ModuleDict()
        self.power_mlp = ModuleDict()

        for node_type in metadata[0]:
            self.lin_node_msg[node_type] = mlp([node_dim[node_type], 16, 32])
            self.lin_node_upd[node_type] = mlp([node_dim[node_type], 16, 32])
            # self.power_mlp[node_type] = mlp([32 + node_dim[node_type], 16])
            self.power_mlp[node_type] = mlp([32, 16])
            self.power_mlp[node_type] = Seq(*[self.power_mlp[node_type], Seq(Lin(16, 1, bias=True), Sigmoid())])

        for edge_type in metadata[1]:
            if self.is_ap_selection:
                self.lin_edge_compact['__'.join(edge_type)] = mlp([edge_dim, 16, 32])
            else:
                self.lin_edge_compact['__'.join(edge_type)] = mlp([edge_dim - 1, 16, 32])

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.power_mlp)
        reset(self.lin_edge_compact)
        reset(self.lin_node_msg)
        reset(self.lin_node_upd)

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Union[Dict[EdgeType, Tensor],
            Dict[EdgeType, SparseTensor]],
            edge_attr_dict: Union[Dict[EdgeType, Tensor],
            Dict[EdgeType, SparseTensor]]
    ) -> tuple[Dict[NodeType, Optional[Tensor]], Union[Dict[EdgeType, Tensor], Dict[EdgeType, SparseTensor]]]:
        out_node_dict = {}

        # Iterate over edge-types to update node_features:
        for edge_type, edge_index in edge_index_dict.items():
            edge_attr = edge_attr_dict[edge_type]
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            x_j = x_dict[src_type]
            x_i = x_dict[dst_type]

            out = self.propagate(edge_index, x=x_j, node_feat=x_i, src_type=src_type, edge_type=edge_type,
                                 dst_type=dst_type, edge_attr=edge_attr, size=(x_j.shape[0], x_i.shape[0]))
            out_node_dict[dst_type] = out
        return out_node_dict, edge_attr_dict

    def message(self, x_j: Tensor, src_type, edge_type, edge_attr) -> Tensor:
        # each node sends its node_ft (both the p_max and the allocated power) and the edge_ft (only the channel)

        node_mlp = self.lin_node_msg[src_type]
        edge_mlp = self.lin_edge_compact[edge_type]
        node_ft = x_j
        if self.is_ap_selection:
            edge_ft = edge_attr
        else:
            edge_ft = edge_attr[:, :1]
        ###
        ap_selection = edge_attr[:, 1].unsqueeze(-1)
        edge_ft = edge_ft * (1 - ap_selection)

        return node_mlp(node_ft) + edge_mlp(edge_ft)

    def update(self, aggr_out, node_feat, dst_type):
        power_max = node_feat[:, 0]
        node_mlp = self.lin_node_upd[dst_type]
        final_mlp = self.power_mlp[dst_type]
        res = node_mlp(node_feat)
        # tmp = torch.cat([node_feat, aggr_out + res], dim=1)
        # power = final_mlp(tmp)
        power = final_mlp(aggr_out + res)
        # return torch.cat([node_feat[:, :-1], power], dim=1)
        if dst_type == 'ue':
            return torch.cat([power_max.unsqueeze(-1), power], dim=1)
        else:
            return power


class EdgeConv(MessagePassing):
    def __init__(self, node_dim: Dict, edge_dim, metadata: Metadata, aggr='mean', **kwargs):
        super(EdgeConv, self).__init__(aggr=aggr)
        all_nod_dim = 0
        for node_type in metadata[0]:
            all_nod_dim = all_nod_dim + node_dim[node_type]
        self.ap_mlp = mlp([5, 16])
        self.ap_mlp = Seq(*[self.ap_mlp, Seq(Lin(16, 1, bias=True), Sigmoid())])

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.ap_mlp)

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Union[Dict[EdgeType, Tensor],
            Dict[EdgeType, SparseTensor]],
            edge_attr_dict: Union[Dict[EdgeType, Tensor],
            Dict[EdgeType, SparseTensor]]
    ) -> tuple[dict[str, Tensor], Union[dict[tuple[str, str, str], Tensor], dict[tuple[str, str, str], SparseTensor]]]:
        edge_dict = edge_attr_dict

        # Iterate over edge-types to update node_features:
        for edge_type, edge_index in edge_index_dict.items():
            edge_attr = edge_attr_dict[edge_type]
            original_edge = edge_type
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            x_j = x_dict[src_type]
            x_i = x_dict[dst_type]
            if edge_type == 'ue__uplink__ap':  # only update edge when msg passing from AP to UE?
                edge_dict[original_edge] = self.edge_updater(edge_index, x=x_j, node_feat=x_i, edge_attr=edge_attr)

        return x_dict, edge_dict

    def edge_update(self, edge_index, x, node_feat, edge_attr):
        unique_values, _, _ = torch.unique(edge_index[0], return_inverse=True, return_counts=True)
        # for each_ue in unique_values:
        #   print(x[each_ue][1])
        src_features = x[edge_index[0]]  # [:,1].unsqueeze(-1)
        dst_features = node_feat[edge_index[1]]  # [:,1].unsqueeze(-1)
        edge_features = edge_attr  # [:,1].unsqueeze(-1)
        tmp = torch.cat([src_features, dst_features, edge_features], dim=1)
        out = self.ap_mlp(tmp)

        ##
        num_ue_all = x.shape[0]
        num_ap_all = node_feat.shape[0]
        num_edge_all = edge_attr.shape[0]
        num_sam = num_ue_all * num_ap_all / num_edge_all

        ## Softmax applying
        set_size = int(num_ap_all / num_sam)
        num_sets = out.shape[0] // set_size
        out_reshaped = out.view(num_sets, set_size, -1)
        softmaxed = torch.softmax(out_reshaped, dim=1)
        max_indices = torch.argmax(softmaxed, dim=1)
        mask = torch.zeros_like(softmaxed)
        mask.scatter_(1, max_indices.unsqueeze(1), 1)
        out = mask.view(-1, 1)

        return torch.cat([edge_attr[:, 0].unsqueeze(-1), out], dim=1)


class HetNetGNN(nn.Module):
    def __init__(self, dataset, num_layers):
        super(HetNetGNN, self).__init__()
        ue2ap_layers, power_layers, ap_selection_layers = num_layers

        self.convs = torch.nn.ModuleList()

        for _ in range(ue2ap_layers):
            conv = Ue2Ap(node_dim={'ue': 2, 'ap': 1}, edge_dim=2,
                         metadata=dataset.metadata(), aggr='mean')
            self.convs.append(conv)

        for _ in range(power_layers):
            conv = PowerConv(node_dim={'ue': 2, 'ap': 1}, edge_dim=2,
                             metadata=dataset.metadata(), aggr='mean')
            self.convs.append(conv)

        for _ in range(ap_selection_layers):
            conv = EdgeConv(node_dim={'ue': 2, 'ap': 1}, edge_dim=2,
                            metadata=dataset.metadata(), aggr='mean')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for conv in self.convs:
            x_dict, edge_attr_dict = conv(x_dict=x_dict, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict)
        return x_dict, edge_attr_dict


class HetNetGNN_v2(nn.Module):
    def __init__(self, dataset, num_layers, ap_selection=False):
        super(HetNetGNN_v2, self).__init__()
        ue2ap_layers, power_layers, ap_selection_layers, final_layers = num_layers

        self.convs = torch.nn.ModuleList()

        for _ in range(ue2ap_layers):
            conv = Ue2Ap(node_dim={'ue': 2, 'ap': 1}, edge_dim=2,
                         metadata=dataset.metadata(), aggr='add')
            self.convs.append(conv)

        for _ in range(power_layers):
            conv = PowerConv(node_dim={'ue': 2, 'ap': 1}, edge_dim=2,
                             metadata=dataset.metadata(), aggr='add')
            self.convs.append(conv)

        for _ in range(ap_selection_layers):
            conv = EdgeConv(node_dim={'ue': 2, 'ap': 1}, edge_dim=2,
                            metadata=dataset.metadata(), aggr='add')
            self.convs.append(conv)

        for _ in range(final_layers):
            conv = PowerConv(node_dim={'ue': 2, 'ap': 1}, edge_dim=2,
                             metadata=dataset.metadata(), ap_selection=ap_selection, aggr='add')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for conv in self.convs:
            x_dict, edge_attr_dict = conv(x_dict=x_dict, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict)
        return x_dict, edge_attr_dict


class HetNetGNN_combine(nn.Module):
    def __init__(self, dataset, num_layers):
        super(HetNetGNN_combine, self).__init__()
        ue2ap_layers, comb_layers = num_layers

        self.convs = torch.nn.ModuleList()

        for _ in range(ue2ap_layers):
            conv = Ue2Ap(node_dim={'ue': 2, 'ap': 1}, edge_dim=2,
                         metadata=dataset.metadata(), aggr='mean')
            self.convs.append(conv)

        for _ in range(comb_layers):
            conv = ResourceConv(node_dim={'ue': 2, 'ap': 1}, edge_dim=2,
                                metadata=dataset.metadata(), ap_selection=False, aggr='mean')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for conv in self.convs:
            x_dict, edge_attr_dict = conv(x_dict=x_dict, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict)
        return x_dict, edge_attr_dict

