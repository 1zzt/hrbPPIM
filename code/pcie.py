
import torch
from torch.nn import Linear
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing


class PCIE(nn.Module):
    def __init__(self, in_channels, out_channels, d_count = 9):
        super(PCIE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_count = d_count

        self.node_mlp = nn.Sequential(
            nn.Linear(self.out_channels * 2, self.out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))
        
        self.node_mlp0 = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))
        
        # self.node_mlp2 = nn.Sequential(
        #     nn.Linear(self.in_channels, self.out_channels),
        #     nn.Dropout(0.1),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm1d(self.out_channels))    
           
        self.edge_row_mlp = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels))

        self.edge_col_mlp = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels))
        
        # self.mlp_rbf_cov = nn.Sequential(nn.Linear(self.d_count, self.in_channels), nn.SiLU())
        self.mlp_rbf_cov = nn.Sequential(nn.Linear(self.d_count, self.out_channels), nn.SiLU())

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        out = x + out
        # out = self.node_mlp2(out)
        return out
    
    def forward(self, x, edge_index_inter, pos=None):
        
        row, col = edge_index_inter
        coord_diff_ncov = pos[row] - pos[col]   # num_edge, 3
        radial_ncov = self.mlp_rbf_cov(_rbf(torch.norm(coord_diff_ncov, dim=-1), D_min=0., D_max=20., D_count=self.d_count, device=x.device))  # num_edge, D_count  --> num_edge, in_channels

        x = self.node_mlp0(x)

        out = torch.cat([x[row], x[col], radial_ncov], dim=1)   # num_edge, x.shape * 3
        
        row_out = self.edge_row_mlp(out)    # num_edge, out_channels
        col_out = self.edge_col_mlp(out.clone())    # num_edge, out_channels
        edge_feat = x[row] * torch.sigmoid(row_out) + x[col] * torch.sigmoid(col_out)

        out_node = self.node_model(x, edge_index_inter, edge_feat)


        return out_node
    


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):

    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def unsorted_segment_mean(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def unsorted_segment_sum(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result



# ---------------------------Run---------------------------------------


# x = torch.rand(7,8)
# edge_index_inter = torch.tensor([[0, 0, 1, 1, 1, 2, 3, 4, 5, 4, 5, 6, 5, 6],
#                     [4, 5, 4, 5, 6, 5, 6, 0, 0, 1, 1, 1, 2, 3]])
# pos = torch.rand(7,3)

# pcie = PCIE(8, 8)

# out = pcie(x, edge_index_inter, pos)