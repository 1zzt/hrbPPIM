
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from torch_geometric.utils import dense_to_sparse





class EnergyDecoder(nn.Module):
    """
    Reference: https://github.com/ACE-KAIST/PIGNet/blob/main/models.py
    """

    def __init__(
        self,
        emb,
        vdw_N=6.0,
        max_vdw_interaction=0.0356,
        min_vdw_interaction=0.0178,
        dev_vdw_radius=0.2,
        no_rotor_penalty=True,
        **kwargs,
    ):
        """Initializes the module
        Args:
            node_in_dim: node dimensions (s, V) in input graph
            node_h_dim: node dimensions to use in GVP-GNN layers
            edge_in_dim: edge dimensions (s, V) in input graph
            edge_h_dim: edge dimensions to embed to before use in GVP-GNN layers
            num_layers: number of GVP-GNN layers
            drop_rate: rate to use in all dropout layers
            residual: whether to have residual connections among GNN layers
            num_outputs: number of output units
            seq_embedding: whether to one-hot embed the sequence
        Returns:
            None
        """
        super(EnergyDecoder, self).__init__()
        self.vdw_N = vdw_N
        self.max_vdw_interaction = max_vdw_interaction
        self.min_vdw_interaction = min_vdw_interaction
        self.dev_vdw_radius = dev_vdw_radius
        self.no_rotor_penalty = no_rotor_penalty

        self.cal_vdw_interaction_A = nn.Sequential(
            nn.Linear(emb, emb),
            nn.ReLU(),
            nn.Linear(emb, 1),
            nn.Sigmoid(),
        )
        self.cal_vdw_interaction_B = nn.Sequential(
            nn.Linear(emb, emb),
            nn.ReLU(),
            nn.Linear(emb, 1),
            nn.Tanh(),
        )
        self.cal_vdw_interaction_N = nn.Sequential(
            nn.Linear(emb, emb),
            nn.ReLU(),
            nn.Linear(emb, 1),
            nn.Sigmoid(),
        )
        self.hbond_coeff = nn.Parameter(torch.tensor([1.0]))
        self.hydrophobic_coeff = nn.Parameter(torch.tensor([0.5]))
        self.vdw_coeff = nn.Parameter(torch.tensor([1.0]))
        self.torsion_coeff = nn.Parameter(torch.tensor([1.0]))
        self.rotor_coeff = nn.Parameter(torch.tensor([0.5]))

    def cal_hbond(
        self,
        dm: Tensor,
        h: Tensor,
        ligand_vdw_radii: Tensor,
        target_vdw_radii: Tensor,
        A: Tensor,
    ) -> Tensor:
        ligand_vdw_radii_ = ligand_vdw_radii.unsqueeze(2).repeat(
            1, 1, target_vdw_radii.size(1)
        )
        target_vdw_radii_ = target_vdw_radii.unsqueeze(1).repeat(
            1, ligand_vdw_radii.size(1), 1
        )
        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.dev_vdw_radius
        dm_0 = ligand_vdw_radii_ + target_vdw_radii_ + B
        dm = dm - dm_0

        # dm: [batch_size, n_nodes_ligand, n_nodes_target]
        retval = dm * A / -0.7
        retval = retval.clamp(min=0.0, max=1.0)
        retval = retval * -(self.hbond_coeff * self.hbond_coeff)
        # retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval

    def cal_hydrophobic(
        self,
        dm: Tensor,
        h: Tensor,
        ligand_vdw_radii: Tensor,
        target_vdw_radii: Tensor,
        A: Tensor,
    ) -> Tensor:
        ligand_vdw_radii_ = ligand_vdw_radii.unsqueeze(2).repeat(
            1, 1, target_vdw_radii.size(1)
        )
        target_vdw_radii_ = target_vdw_radii.unsqueeze(1).repeat(
            1, ligand_vdw_radii.size(1), 1
        )
        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.dev_vdw_radius
        dm_0 = ligand_vdw_radii_ + target_vdw_radii_ + B
        dm = dm - dm_0

        retval = (-dm + 1.5) * A
        retval = retval.clamp(min=0.0, max=1.0)
        retval = retval * -(self.hydrophobic_coeff * self.hydrophobic_coeff)
        # retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval

    def cal_vdw_interaction(
        self,
        dm: Tensor,
        h: Tensor,
        ligand_vdw_radii: Tensor,
        target_vdw_radii: Tensor,
        ligand_valid: Tensor,
        target_valid: Tensor,
    ) -> Tensor:
        ligand_valid_ = ligand_valid.unsqueeze(2).repeat(
            1, 1, target_valid.size(1)
        )
        target_valid_ = target_valid.unsqueeze(1).repeat(
            1, ligand_valid.size(1), 1
        )
        ligand_vdw_radii_ = ligand_vdw_radii.unsqueeze(2).repeat(
            1, 1, target_vdw_radii.size(1)
        )
        target_vdw_radii_ = target_vdw_radii.unsqueeze(1).repeat(
            1, ligand_vdw_radii.size(1), 1
        )

        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.dev_vdw_radius
        dm_0 = ligand_vdw_radii_ + target_vdw_radii_ + B
        dm_0[dm_0 < 0.0001] = 1
        N = self.vdw_N
        vdw_term1 = torch.pow(dm_0 / dm, 2 * N)
        vdw_term2 = -2 * torch.pow(dm_0 / dm, N)

        A = self.cal_vdw_interaction_A(h).squeeze(-1)
        A = A * (self.max_vdw_interaction - self.min_vdw_interaction)
        A = A + self.min_vdw_interaction

        energy = vdw_term1 + vdw_term2
        energy = energy.clamp(max=100)
        energy = energy * ligand_valid_ * target_valid_
        energy = A * energy
        # energy = energy.sum(1).sum(1).unsqueeze(-1)
        return energy

    def cal_distance_matrix(
        self, ligand_pos: Tensor, target_pos: Tensor, dm_min: float
    ) -> Tensor:
        p1_repeat = ligand_pos.unsqueeze(2).repeat(1, 1, target_pos.size(1), 1)
        p2_repeat = target_pos.unsqueeze(1).repeat(1, ligand_pos.size(1), 1, 1)
        dm = torch.sqrt(torch.pow(p1_repeat - p2_repeat, 2).sum(-1) + 1e-10)
        replace_vec = torch.ones_like(dm) * 1e10
        dm = torch.where(dm < dm_min, replace_vec, dm)
        return dm

    def forward(self, 
            edge_index_hbond,
            edge_index_metal,
            edge_index_hydrophobic,
            ligand_pos,
            target_pos,
            # rotor,
            ligand_vdw_radii,
            target_vdw_radii,
            ligand_non_metal,
            target_non_metal,
            h_cat, 
            DM_min=0.5, 
            cal_der_loss=False):
        """Perform the forward pass.
        Args:
            sample: List of tensors created by pignet_featurizers
            h_cat: torch.Tensor batch of hidden representations of atoms
        Returns:
            energies, der1, der2
        """
        
        # distance matrix
        # ligand_pos.requires_grad = True
        dm = self.cal_distance_matrix(
            ligand_pos, target_pos, DM_min
        )  # [batch_size, n_nodes_ligand, n_nodes_target]

        # compute energy component
        energies = []

        # vdw interaction
        vdw_energy = self.cal_vdw_interaction(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            ligand_non_metal,
            target_non_metal,
        )
        energies.append(vdw_energy)

        # hbond interaction
        hbond = self.cal_hbond(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            edge_index_hbond,
        )
        # energies.append(hbond)

        # metal interaction
        metal = self.cal_hbond(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            edge_index_metal
        )
        # energies.append(metal)

        # hydrophobic interaction
        hydrophobic = self.cal_hydrophobic(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            edge_index_hydrophobic,
        )
        # energies.append(hydrophobic)

        # energies = torch.cat(energies, -1)
        energies = torch.stack([vdw_energy, hbond, metal, hydrophobic], dim=-1)
        # energies = torch.stack([vdw_energy, hbond, metal, hydrophobic], dim=-1)


        # rotor penalty
        if not self.no_rotor_penalty:
            energies = energies / (
                1 + self.rotor_coeff * self.rotor_coeff * rotor.unsqueeze(-1)
            )

        # derivatives
        if cal_der_loss:
            gradient = torch.autograd.grad(
                energies.sum(),
                ligand_pos,
                retain_graph=True,
                create_graph=True,
            )[0]
            der1 = torch.pow(gradient.sum(1), 2).mean()
            der2 = torch.autograd.grad(
                gradient.sum(),
                ligand_pos,
                retain_graph=True,
                create_graph=True,
            )[0]
            der2 = -der2.sum(1).sum(1).mean()
        else:
            der1 = torch.zeros_like(energies).sum()
            der2 = torch.zeros_like(energies).sum()

        return energies, der1, der2
