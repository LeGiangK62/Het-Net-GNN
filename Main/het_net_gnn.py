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


class Ue2Ap(MessagePassing):
    def __init__(self, node_dim: Dict, edge_dims: Dict, out_node_dim, metadata: Metadata, aggr='mean'):
        super(Ue2Ap, self).__init__(aggr=aggr)
        self.edge_in_dim = edge_dims['in']
        self.edge_out_dim = edge_dims['out']
        self.lin_node_msg_compact = ModuleDict()
        self.lin_edge_compact = ModuleDict()
        self.lin_node_upd = ModuleDict()
        out_dim_src = {'ue': out_node_dim,
                       'ap': out_node_dim - 1}
        for node_type in metadata[0]:
            self.lin_node_upd[node_type] = mlp([node_dim[node_type], 16, out_node_dim - 1])
            if node_dim[node_type]:
                self.lin_node_msg_compact[node_type] = mlp([node_dim[node_type], 16, out_dim_src[node_type]])
        for edge_type in metadata[1]:
            src_type, _, dst_type = edge_type
            self.lin_edge_compact['__'.join(edge_type)] = mlp([self.edge_in_dim - 1, 16, out_dim_src[src_type]])
        self.edge_mlp = mlp([out_node_dim * 2 + self.edge_in_dim - 1, 16, self.edge_out_dim - 2])

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.lin_edge_compact)
        reset(self.lin_node_msg_compact)
        reset(self.lin_node_upd)
        reset(self.edge_mlp)

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Union[Dict[EdgeType, Tensor],
            Dict[EdgeType, SparseTensor]],
            edge_attr_dict: Union[Dict[EdgeType, Tensor],
            Dict[EdgeType, SparseTensor]]
    ) -> tuple[Dict[NodeType, Optional[Tensor]], Union[Dict[EdgeType, Tensor], Dict[EdgeType, SparseTensor]]]:
        out_node_dict = {}

        edge_dict = edge_attr_dict

        # Iterate over edge-types to update node_features:
        for edge_type, edge_index in edge_index_dict.items():
            edge_attr = edge_attr_dict[edge_type]
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            x_j = x_dict[src_type]
            x_i = x_dict[dst_type]
            out_node_dict[dst_type] = self.propagate(edge_index, x=x_j, node_feat=x_i, src_type=src_type,
                                                     edge_type=edge_type,
                                                     dst_type=dst_type, edge_attr=edge_attr,
                                                     size=(x_j.shape[0], x_i.shape[0]))

        for edge_type, edge_index in edge_index_dict.items():
            edge_attr = edge_attr_dict[edge_type]
            original_edge = edge_type
            src_type, _, dst_type = edge_type
            x_j = out_node_dict[src_type]
            x_i = out_node_dict[dst_type]
            edge_dict[original_edge] = self.edge_updater(edge_index, x=x_j, node_feat=x_i, edge_attr=edge_attr,
                                                         src_type=src_type)

        return out_node_dict, edge_attr_dict

    def message(self, x_j: Tensor, node_feat: Tensor, src_type, edge_type, edge_attr) -> Tensor:
        # APs only send the edge_feat (only the channel) to UEs
        # Ues send the node_ft (only the p_max) and the edge_ft (only the channel)
        edge_mlp = self.lin_edge_compact[edge_type]
        edge_ft = edge_attr[:, :1]
        if src_type == 'ue':
            node_ft = x_j
            node_mlp = self.lin_node_msg_compact[src_type]
            return node_mlp(node_ft) + edge_mlp(edge_ft)
        else:
            return edge_mlp(edge_ft)

    def edge_update(self, edge_index, x, node_feat, edge_attr, src_type):
        # Adding this to unify uplink and downlink edges
        if src_type == 'ue':
            ue_features = x[edge_index[0]]
            ap_features = node_feat[edge_index[1]]
        else:
            ap_features = x[edge_index[0]]
            ue_features = node_feat[edge_index[1]]
        edge_features = edge_attr[:, :-1]
        tmp = torch.cat([ue_features, ap_features, edge_features], dim=1)

        if self.edge_in_dim != self.edge_out_dim:
            expanded = self.edge_mlp(tmp)
            return torch.cat([edge_attr[:, 0].unsqueeze(-1), expanded, edge_attr[:, -1].unsqueeze(-1)], dim=1)
        else:
            return edge_attr

    def update(self, aggr_out, node_feat, dst_type):
        if dst_type == 'ue':
            node_mlp = self.lin_node_upd[dst_type]
            res = node_mlp(node_feat)
            new_node_feat = aggr_out + res
            return torch.cat([node_feat[:, :1], new_node_feat], dim=1)
        else:
            return aggr_out


class EdgeConv(MessagePassing):
    def __init__(self, node_dim: Dict, edge_dim, out_node_dim, metadata: Metadata, aggr='mean', **kwargs):
        super(EdgeConv, self).__init__(aggr=aggr)
        all_node_dim = sum(node_dim.values())

        self.ap_mlp = mlp([all_node_dim + edge_dim - 1, 16])
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
    ) -> tuple[Dict[NodeType, Optional[Tensor]], Union[Dict[EdgeType, Tensor], Dict[EdgeType, SparseTensor]]]:
        out_node_dict = x_dict
        edge_dict = edge_attr_dict

        for edge_type, edge_index in edge_index_dict.items():
            edge_attr = edge_attr_dict[edge_type]
            original_edge = edge_type
            src_type, _, dst_type = edge_type
            x_j = out_node_dict[src_type]
            x_i = out_node_dict[dst_type]

            edge_dict[original_edge] = self.edge_updater(edge_index, x=x_j, node_feat=x_i,
                                                         edge_attr=edge_attr, src_type=src_type)
        return out_node_dict, edge_dict

    def edge_update(self, edge_index, x, node_feat, edge_attr, src_type):
        unique_values, _, _ = torch.unique(edge_index[0], return_inverse=True, return_counts=True)

        if src_type == 'ue':
            ue_features = x[edge_index[0]]
            ap_features = node_feat[edge_index[1]]
        else:
            ap_features = x[edge_index[0]]
            ue_features = node_feat[edge_index[1]]
        edge_features = edge_attr[:, :-1]
        tmp = torch.cat([ue_features, ap_features, edge_features], dim=1)
        out = self.ap_mlp(tmp)
        #
        # Align AP_selection
        num_edge_all = edge_attr.shape[0]

        if src_type == 'ue':
            num_ue_all = x.shape[0]
            num_ap_all = node_feat.shape[0]
            num_sam = int(num_ue_all * num_ap_all / num_edge_all)

            num_ap = int(num_ap_all / num_sam)
            num_ue = int(num_ue_all / num_sam)

            out_reshaped = out.view(-1, num_ap)
        else:
            num_ap_all = x.shape[0]
            num_ue_all = node_feat.shape[0]
            num_sam = int(num_ue_all * num_ap_all / num_edge_all)

            num_ap = int(num_ap_all / num_sam)
            num_ue = int(num_ue_all / num_sam)

            out_reshaped = out.view(-1, num_ap, num_ue).permute(0, 2, 1).reshape(-1, num_ap)
        # Softmax applying
        out = torch.softmax(out_reshaped, dim=1)
        if src_type == 'ue':
            out = out.view(-1, 1)

        else:
            out = out.view(-1, num_ue, num_ap).permute(0, 2, 1).reshape(-1, 1)

        return torch.cat([edge_attr[:, :-1], out], dim=1)


class PowerConv(MessagePassing):
    def __init__(self, node_dim: Dict, edge_dim, out_node_dim, metadata: Metadata, aggr='mean', **kwargs):
        super(PowerConv, self).__init__(aggr=aggr)
        self.lin_node_msg = ModuleDict()
        self.lin_node_upd = ModuleDict()
        self.msg_mlp = ModuleDict()
        all_node_dim = sum(node_dim.values())
        out_dim_dst = {'ap': out_node_dim,
                       'ue': out_node_dim - 1}
        out_dim_src = {'ue': out_node_dim,
                       'ap': out_node_dim - 1}
        for node_type in metadata[0]:
            self.lin_node_msg[node_type] = mlp([node_dim[node_type], 16, out_node_dim - 1])
            self.lin_node_upd[node_type] = mlp([node_dim[node_type], 16, out_dim_dst[node_type]])
            self.msg_mlp[node_type] = mlp([all_node_dim + edge_dim - 1, 16, out_dim_src[node_type]])
        # self.edge_mlp = mlp([all_node_dim + edge_dim - 1, 16, 1])

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.lin_node_msg)
        reset(self.lin_node_upd)
        reset(self.msg_mlp)

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Union[Dict[EdgeType, Tensor],
            Dict[EdgeType, SparseTensor]],
            edge_attr_dict: Union[Dict[EdgeType, Tensor],
            Dict[EdgeType, SparseTensor]]
    ) -> tuple[Dict[NodeType, Optional[Tensor]], Union[Dict[EdgeType, Tensor], Dict[EdgeType, SparseTensor]]]:
        out_node_dict = {}
        edge_dict = edge_attr_dict

        # Iterate over edge-types to update node_features:
        for edge_type, edge_index in edge_index_dict.items():
            edge_attr = edge_attr_dict[edge_type]
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            x_j = x_dict[src_type]
            x_i = x_dict[dst_type]
            out_node_dict[dst_type] = self.propagate(edge_index, x=x_j, node_feat=x_i, src_type=src_type,
                                                     edge_type=edge_type,
                                                     dst_type=dst_type, edge_attr=edge_attr,
                                                     size=(x_j.shape[0], x_i.shape[0]))

        return out_node_dict, edge_dict

    def message(self, x_j: Tensor, node_feat, src_type, edge_type, edge_attr) -> Tensor:
        # each node sends its node_ft (both the p_max and the allocated power) and the edge_ft (only the channel)
        dst_feat = node_feat.repeat(x_j.shape[0] // node_feat.shape[0], 1)
        edge_ft = edge_attr[:, :-1]
        if src_type == 'ap':
            # APs do not need the ap_selection information
            ap_selection = edge_attr[:, -1].unsqueeze(-1)
            edge_ft = edge_ft * ap_selection

        tmp_msg = torch.cat([x_j, edge_ft, dst_feat], dim=1)
        message_mlp = self.msg_mlp[src_type]

        return message_mlp(tmp_msg)

    def update(self, aggr_out, node_feat, dst_type):
        node_mlp = self.lin_node_upd[dst_type]
        res = node_mlp(node_feat)
        new_node_feat = aggr_out + res
        if dst_type == 'ue':
            return torch.cat([node_feat[:, :1], new_node_feat], dim=1)
        else:
            return new_node_feat
