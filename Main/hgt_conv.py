import torch
from torch_geometric.nn import Linear, HGTConv



#region Build Heterogeneous GNN
class HGTGNN(torch.nn.Module):
    def __init__(self, data, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

        self.lin1 = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        original = x_dict['ue'].clone
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        original = x_dict['ue']  # not original
        power = self.lin(x_dict['ue'])
        ap_selection = self.lin1(x_dict['ue'])
        ap_selection = torch.abs(ap_selection).int()

        out = torch.cat((original[:, 1].unsqueeze(-1), power[:, 1].unsqueeze(-1), ap_selection[:, 1].unsqueeze(-1)), 1)
        return out
#endregion

