from typing import Dict, List, Optional, Union

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



def mlp(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i], bias=True), ReLU())  # , BN(channels[i]))
        for i in range(1, len(channels))
    ])


class EdgeConv(MessagePassing):
    def __init__(self, node_dim, edge_dim, metadata: Metadata, aggr='mean', **kwargs):
        super(EdgeConv, self).__init__(aggr=aggr)

        self.lin_node = ModuleDict()
        self.lin_edge = ModuleDict()
        self.res_lin = ModuleDict()

        for node_type in metadata[0]:
            self.lin_node[node_type] = mlp([node_dim, 32])
            self.res_lin[node_type] = nn.Linear(node_dim, 32)

        for edge_type in metadata[1]:
            self.lin_edge['__'.join(edge_type)] = mlp([edge_dim, 32])

        self.power_mlp = mlp([32 + node_dim, 16])
        self.power_mlp = Seq(*[self.power_mlp, Seq(Lin(16, 1, bias=True), Sigmoid())])
        self.ap_mlp = mlp([32 + node_dim, 16])
        self.ap_mlp = Seq(*[self.ap_mlp, Seq(Lin(16, 1, bias=True), Sigmoid())])


        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.lin_node)
        reset(self.lin_edge)
        reset(self.res_lin)
        reset(self.power_mlp)
        reset(self.ap_mlp)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Union[Dict[EdgeType, Tensor],
                               Dict[EdgeType, SparseTensor]],  # Support both.
        edge_attr_dict: Union[Dict[EdgeType, Tensor],
                              Dict[EdgeType, SparseTensor]]
    ) -> Dict[NodeType, Optional[Tensor]]:
        # How to get the edge attributes from only the index?
        lin_node_dict, lin_edge_dict, out_dict = {}, {}, {}
        # Iterate over node-types to initialize the output dictionary
        for node_type, node_ft in x_dict.items():
            lin_node_dict[node_type] = self.lin_node[node_type](node_ft)
            out_dict[node_type] = []

        # Iterate over edge-types:
        for edge_type, edge_index in edge_index_dict.items():
            # aggregate information to the destination node
            # for each type of edge
            edge_attr = edge_attr_dict[edge_type]
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            x_j = x_dict[src_type]
            x_i = x_dict[dst_type]

            out = self.propagate(edge_index, x=x_j, node_feat=x_i, src_type=src_type, edge_type=edge_type,
                                 dst_type=dst_type, edge_attr=edge_attr, size=(x_j.shape[0], x_i.shape[0]))
            out_dict[dst_type] = out

        # Iterate over node-types:
        # for node_type, outs in out_dict.items():
            # out = group(outs, self.group)
            #
            # if out is None:
            #     out_dict[node_type] = None
            #     continue
            #
            # out = self.a_lin[node_type](F.gelu(out))
            # if out.size(-1) == x_dict[node_type].size(-1):
            #     alpha = self.skip[node_type].sigmoid()
            #     out = alpha * out + (1 - alpha) * x_dict[node_type]
            # out_dict[node_type] = out

        return out_dict

    # def message(self, x_j: Tensor, src_type, edge_type, node_mlp, edge_rel) -> Tensor:
    def message(self, x_j: Tensor, src_type, edge_type, edge_attr) -> Tensor:
        # This function is called when we use self.propagate - Used the given parameters too.
        # What each neighbor node send to target along the edges
        # Adding the edge relation here
        node_mlp = self.lin_node[src_type]
        edge_mlp = self.lin_edge[edge_type]
        return node_mlp(x_j) + edge_mlp(edge_attr)

    def update(self, aggr_out, node_feat, dst_type, edge_index):
        # Update node representations with the aggregated messages
        # aggr_out  = output of aggregation function, the following is the input of the propagation function
        power_max = node_feat[:, 0]
        # ap_selection = node_feat[:, 2].unsqueeze(-1)
        node_mlp = self.lin_node[dst_type]
        res = node_mlp(node_feat)
        tmp = torch.cat([node_feat, aggr_out + res], dim=1)
        power = self.power_mlp(tmp)
        # ap_selection = self.ap_mlp(tmp)
        return torch.cat([power_max.unsqueeze(-1), power], dim=1)


class RGCN(nn.Module):
    def __init__(self, dataset, num_layers):
        # The only things need to fix here are the dimensions
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.mlp = mlp([32, 16])
        for _ in range(num_layers):
            conv = EdgeConv(node_dim=2, edge_dim=2,
                            metadata=dataset.metadata())
            self.convs.append(conv)


    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for conv in self.convs:
            x_dict = conv(x_dict=x_dict, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict)
            # x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict

