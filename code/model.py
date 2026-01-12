
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (MessagePassing)
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from pcie import PCIE
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import DenseGCNConv
from torch_geometric.nn.conv import MessagePassing
import math
from torch.nn import Linear
from energy import EnergyDecoder


class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)

        return h
    
class GINConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__(aggr=aggr)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)


        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out



class GNNComplete(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0., gnn_type="gin"):

        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        super(GNNComplete, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layer = num_layer
        self.JK = JK
        # self.fc = FC(emb_dim * 3, emb_dim, 3, 0.1, 1)

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim, aggr="add"))

        ###List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")
        x = self.atom_encoder(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("not implemented.")
        

        return node_representation


def gnn_norm(x, norm):

    batch_size, num_nodes, num_channels = x.size()
    x = x.view(-1, num_channels)    # torch.Size([b*num_nodes, num_red_nodes])
    x = norm(x) # torch.Size([b*num_nodes, num_red_nodes])
    x = x.view(batch_size, num_nodes, num_channels)

    return x
   
class DiffPool(nn.Module):
    def __init__(self, input_dim, output_dim, max_num, red_node):
        super().__init__()

        self.max_num = max_num
        self.red_node = red_node
        self.gnn_p = DenseGCNConv(input_dim, red_node, improved=True, bias=True)
        self.gnn_p_norm = nn.Sequential(
            nn.BatchNorm1d(red_node),
            nn.Mish(),
        )
        self.gnn_e = DenseGCNConv(input_dim, output_dim, improved=True, bias=True)
        self.gnn_e_norm = nn.Sequential(
            nn.BatchNorm1d(output_dim),
            nn.Mish(),
        )
        self.out = nn.Linear(output_dim, output_dim)
        self.out_norm = nn.Sequential(
            nn.BatchNorm1d(output_dim),
        )

    def pooling(self, x, adj, s, mask=None):

        batch_size, num_nodes, _ = x.size()
        x = x.unsqueeze(0) if x.dim() == 2 else x   # torch.Size([b, 600, 300])
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj   # torch.Size([8, 600, 600])
        s = s.unsqueeze(0) if s.dim() == 2 else s   # torch.Size([8, 600, 28])
        s = F.softmax(s, dim=-1)    # torch.Size([8, 600, 28])  

        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)  # torch.Size([8, 600, 1])
            x, s = x * mask, s * mask

        out = torch.matmul(s.transpose(1, 2), x)    # torch.Size([8, 28, 300]) sT: [b, block, num_nodes] @ x: [b, num_nodes, dim]
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)     
        
        return out, out_adj, s
    

    def forward(self, x, adj, mask):

        gnn_out = self.gnn_p(x, adj, mask)  # gnn_out: [b, max_num_nodes, red_node]
        s = gnn_norm(gnn_out, self.gnn_p_norm) # torch.Size([b, max_num_nodes, red_node])
        x, adj, s = self.pooling(x, adj, s, mask)   
        x = gnn_norm(self.gnn_e(x, adj), self.gnn_e_norm)
        x = gnn_norm(self.out(x), self.out_norm)

        return x, s

   
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(MLP, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Mish(),
            nn.Dropout(drop_rate),
        )
        
    def forward(self, x):
        
        return self.mlp(x)
    

    
class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, heads, drop_rate, ca=True):
        super().__init__()

        self.heads = heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // heads
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = MLP(hidden_dim, hidden_dim, drop_rate)
        self.W_P = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.ca = ca
        
    def forward(self, q, k, v, attn_bias = None): 

        

        batch_size, seqlen_q, _ = q.shape
        _, seqlen_k, _ = k.shape
        
        Q = self.W_Q(q) # [batch_size, seqlen_q, hidden_dim]
        K = self.W_K(k)
        V = self.W_V(v)
        
        Q = Q.view(batch_size, seqlen_q, self.heads, self.head_dim).transpose(1, 2)  # [batch_size, heads, seqlen_q, head_dim]
        K = K.view(batch_size, seqlen_k, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seqlen_k, self.heads, self.head_dim).transpose(1, 2)
        
        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_bias is not None:
            energy = energy + attn_bias
        attention = torch.softmax(energy, dim=-1)  # [batch_size, num_heads, seqlen_q, seqlen_k]
        x = torch.matmul(attention, V)  # [batch_size, num_heads, seqlen_q, head_dim]
        x = x.transpose(1, 2).contiguous().view(batch_size, seqlen_q, self.hidden_dim)  # [batch_size, seqlen_q, hidden_dim]
        
        if self.ca:
            res = q.sum(dim=1)

            x = x.sum(dim=1)
            x = self.W_O(x) + res
        
        else:
            x = self.W_P(x) + q
        
        return x, attention
    




    
def conv2d(in_channels: int, 
            out_channels: int, 
            kernel_size: int, 
            stride: int = 1, 
            padding: str = "same", 
            dilation: int = 1, 
            group: int = 1, 
            bias: bool = False) -> nn.Conv2d:

    if padding == "same":
        padding = int((kernel_size - 1)/2)

    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias)

def conv_identity_2d( in_channels  : int,
                      out_channels : int,
                      kernel_size  : int = 1,
                      stride       : int = 1,
                      padding      : str = "same",
                      dilation     : int = 1,
                      group        : int = 1,
                      bias         : bool = False,
                      norm         : str = "IN",
                      activation   : str = "Relu",
                      track_running_stats_ : bool = True):
    layers = []

    # convolution
    layers.append(conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias))

    # normalization
    if norm == "BN":
       layers.append( nn.BatchNorm2d(out_channels, affine=True, track_running_stats=track_running_stats_))
    elif norm == "IN":
        layers.append( nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=track_running_stats_))

    # activation
    if activation == "ELU":
        layers.append( nn.ELU())
    elif activation == "Relu":
        layers.append( nn.LeakyReLU(negative_slope=0.01,inplace=True))

    return nn.Sequential(*layers)


class GPANN(nn.Module):
    def __init__(self, emb = 256, heads = 4):
        super(GPANN, self).__init__()
        self.covalentconv = GNNComplete(3, emb, 'last', drop_ratio=0, gnn_type='gin')

        self.pcie = PCIE(57, emb)


        self.layernorm = torch.nn.LayerNorm(heads)
        self.layernorm2 = torch.nn.LayerNorm(heads)


        self.identity = conv_identity_2d(heads, heads, 1, 1, bias=False)
        self.identity1 = conv_identity_2d(emb, emb // 2, 1, 1, bias=False)
        self.identity2 = conv_identity_2d(emb // 2, heads, 1, 1, bias=False)


        self.identityppi = conv_identity_2d(4, heads, 1, 1, bias=False)
        self.identityppitest = conv_identity_2d(1, heads, 1, 1, bias=False)
        self.identityppi1 = conv_identity_2d(emb, emb // 2, 1, 1, bias=False)
        self.identityppi2 = conv_identity_2d(emb // 2, heads, 1, 1, bias=False)

        self.energy = EnergyDecoder(emb = heads)


        self.ligandfeat_linear = Linear(emb, heads)
        self.ppifeat_linear = Linear(emb, heads)


        self.diffpool1 = DiffPool(emb, emb, 600, 28)
        self.diffpool2 = DiffPool(emb, emb, 600, 156)
        

        self.attblock_ppi = AttentionBlock(emb, heads, 0.1, ca=False)
        self.attblock1 = AttentionBlock(emb, heads, 0.1)
        self.attblock2 = AttentionBlock(emb, heads, 0.1)


        self.gate_linear = Linear(heads, heads)

        self.fc = FC(emb, emb, 3, 0.1, 1)

    def forward(self, data):
        
        x_ligand = data['ligand'].x
        edge_index_ligand = data['ligand', 'l2l', 'ligand'].edge_index
        edge_attr_ligand = data['ligand', 'l2l', 'ligand'].edge_attr


        x_ppi = data['protein'].x
        pos_ppi = data['protein'].pos
        

        noncov_edge_index_ppi = data['protein', 'p2p_noncov', 'protein'].edge_index
        alledges_edge_index_ppi = data['protein', 'p2p_alledges', 'protein'].edge_index
        

        x_ligand_cov = self.covalentconv(x_ligand, edge_index_ligand, edge_attr_ligand)


        x_ppi_alledges = self.pcie(x_ppi, alledges_edge_index_ppi, pos_ppi)


        compound_out_batched, compound_out_mask = to_dense_batch(x_ligand_cov, data['ligand'].batch)
        protein_out_batched, protein_out_mask = to_dense_batch(x_ppi_alledges, data['protein'].batch)

        compound_adj = to_dense_adj(edge_index_ligand, data['ligand'].batch)    # [b, n, n]
        protein_adj = to_dense_adj(noncov_edge_index_ppi, data['protein'].batch) # # [b, m, m]

        zligand_feat_in = self.layernorm(self.ligandfeat_linear(compound_out_batched))
        zppi_feat_in = self.layernorm(self.ppifeat_linear(protein_out_batched))


        z = torch.einsum("bik,bjk->bijk", zppi_feat_in, zligand_feat_in)    # torch.Size([B, prot_atom_num, comp_atom_num, 300])
        z_mask = torch.einsum("bi,bj->bij", protein_out_mask, compound_out_mask)    # torch.Size([B, prot_atom_num, comp_atom_num])
        z = z * z_mask.unsqueeze(-1)


        z_final = z.permute(0, 3, 1, 2) + self.identity(z.permute(0, 3, 1, 2))
        z_final = self.gate_linear(z_final.permute(0, 2, 3, 1)).sigmoid() * z_final.permute(0, 2, 3, 1)   # Gated

        zppi_feat_in = self.ppifeat_linear(protein_out_batched)

        zppi_feat = torch.einsum("bik,bjk->bijk", self.layernorm2(zppi_feat_in), self.layernorm2(zppi_feat_in)) 
        zppi_mask = torch.einsum("bi,bj->bij", protein_out_mask, protein_out_mask)

        zppi_feat = zppi_feat * zppi_mask.unsqueeze(-1)

        edge_index_hbond = data['protein', 'hbond', 'protein'].edge_index
        edge_index_metal = data['protein', 'metal', 'protein'].edge_index
        edge_index_hydrophobic = data['protein', 'hydrophobic', 'protein'].edge_index
        
        ligand_vdw_radii = data['protein'].ligand_vdw_radii
        target_vdw_radii = data['protein'].target_vdw_radii
        ligand_non_metal = data['protein'].ligand_non_metal
        target_non_metal = data['protein'].target_non_metal


        edge_index_hbond = to_dense_adj(edge_index_hbond, data['protein'].batch)
        edge_index_metal = to_dense_adj(edge_index_metal, data['protein'].batch)
        edge_index_hydrophobic = to_dense_adj(edge_index_hydrophobic, data['protein'].batch)

        ligand_pos, _ = to_dense_batch(pos_ppi, data['protein'].batch)
        target_pos = ligand_pos

        ligand_vdw_radii, _ = to_dense_batch(ligand_vdw_radii, data['protein'].batch)
        target_vdw_radii = ligand_vdw_radii

        ligand_non_metal, _ = to_dense_batch(ligand_non_metal, data['protein'].batch)
        target_non_metal = ligand_non_metal



        zppi_feat, _, _ = self.energy(
            edge_index_hbond,
            edge_index_metal,
            edge_index_hydrophobic,
            ligand_pos,
            target_pos,
            ligand_vdw_radii,
            target_vdw_radii,
            ligand_non_metal,
            target_non_metal,
            zppi_feat
            )

        zppi_feat = zppi_feat.sum(-1).unsqueeze(-1)
        zppi = self.identityppitest(zppi_feat.permute(0, 3, 1, 2))
        zppi = zppi.permute(0, 2, 3, 1)


        x_lig, s_lig = self.diffpool1(compound_out_batched, compound_adj, compound_out_mask)    # torch.Size([b, 28, 300])
        x_prot, s_prot = self.diffpool2(protein_out_batched, protein_adj, protein_out_mask)

        z1 = torch.einsum('bij,bjkl->bikl', s_prot.transpose(1, 2), z_final)
        z_attn = torch.einsum('bijl,bjk->bikl', z1, s_lig)  # torch.Size([b, prot_num, lig_num, heads])

        z_attn = z_attn.permute(0, 3, 1, 2)  # [b, h, m, n]

        zppi1 = torch.einsum('bij,bjkl->bikl', s_prot.transpose(1, 2), zppi)
        zppi_attn = torch.einsum('bijl,bjk->bikl', zppi1, s_prot).permute(0, 3, 1, 2) 


        x_prot, _ = self.attblock_ppi(x_prot, x_prot, x_prot, zppi_attn)
        
        l2p, _ = self.attblock1(x_lig, x_prot, x_prot, z_attn.permute(0, 1, 3, 2))  # [b, 300]
        p2l, _ = self.attblock2(x_prot, x_lig, x_lig, z_attn)   # [b, 300]

        
        h = l2p + p2l

        h = self.fc(h)

        return h.view(-1)