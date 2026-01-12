
import os
import pickle
from itertools import chain, repeat
from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd
import torch
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from scipy.spatial import distance
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds

from rdkit.Chem import Atom, Mol
from typing import Any, Dict, List
from torch_geometric.utils import dense_to_sparse

HBOND_DONOR_INDICES = ["[!#6;!H0]"]
HBOND_ACCEPPTOR_SMARTS = [
    "[$([!#6;+0]);!$([F,Cl,Br,I]);!$([o,s,nX3]);!$([Nv5,Pv5,Sv4,Sv6])]"
]



def get_hbond_atom_indices(mol: Mol, smarts_list: List[str]) -> np.ndarray:
    indice = []
    for smarts in smarts_list:
        smarts = Chem.MolFromSmarts(smarts)
        indice += [idx[0] for idx in mol.GetSubstructMatches(smarts)]
    indice = np.array(indice)
    return indice


def get_A_hbond(ligand_mol: Mol, target_mol: Mol) -> np.ndarray:
    ligand_h_acc_indice = get_hbond_atom_indices(
        ligand_mol, HBOND_ACCEPPTOR_SMARTS
    )
    target_h_acc_indice = get_hbond_atom_indices(
        target_mol, HBOND_ACCEPPTOR_SMARTS
    )
    ligand_h_donor_indice = get_hbond_atom_indices(
        ligand_mol, HBOND_DONOR_INDICES
    )
    target_h_donor_indice = get_hbond_atom_indices(
        target_mol, HBOND_DONOR_INDICES
    )

    hbond_indice = np.zeros(
        (ligand_mol.GetNumAtoms(), target_mol.GetNumAtoms())
    )
    for i in ligand_h_acc_indice:
        for j in target_h_donor_indice:
            hbond_indice[i, j] = 1
    for i in ligand_h_donor_indice:
        for j in target_h_acc_indice:
            hbond_indice[i, j] = 1
    return hbond_indice

METALS = ("Zn", "Mn", "Co", "Mg", "Ni", "Fe", "Ca", "Cu")
def get_A_metal_complexes(ligand_mol: Mol, target_mol: Mol) -> np.ndarray:
    ligand_h_acc_indice = get_hbond_atom_indices(
        ligand_mol, HBOND_ACCEPPTOR_SMARTS
    )
    target_h_acc_indice = get_hbond_atom_indices(
        target_mol, HBOND_ACCEPPTOR_SMARTS
    )
    ligand_metal_indice = np.array(
        [
            idx
            for idx in range(ligand_mol.GetNumAtoms())
            if ligand_mol.GetAtomWithIdx(idx).GetSymbol() in METALS
        ]
    )
    target_metal_indice = np.array(
        [
            idx
            for idx in range(target_mol.GetNumAtoms())
            if target_mol.GetAtomWithIdx(idx).GetSymbol() in METALS
        ]
    )

    metal_indice = np.zeros(
        (ligand_mol.GetNumAtoms(), target_mol.GetNumAtoms())
    )
    for ligand_idx in ligand_h_acc_indice:
        for target_idx in target_metal_indice:
            metal_indice[ligand_idx, target_idx] = 1
    for ligand_idx in ligand_metal_indice:
        for target_idx in target_h_acc_indice:
            metal_indice[ligand_idx, target_idx] = 1
    return metal_indice

HYDROPHOBICS = ("F", "CL", "BR", "I")

def get_hydrophobic_atom(mol: Mol) -> np.ndarray:
    natoms = mol.GetNumAtoms()
    hydrophobic_indice = np.zeros((natoms,))
    for atom_idx in range(natoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        if symbol.upper() in HYDROPHOBICS:
            hydrophobic_indice[atom_idx] = 1
        elif symbol.upper() in ["C"]:
            neighbors = [x.GetSymbol() for x in atom.GetNeighbors()]
            neighbors_wo_c = list(set(neighbors) - set(["C"]))
            if len(neighbors_wo_c) == 0:
                hydrophobic_indice[atom_idx] = 1
    return hydrophobic_indice

def get_A_hydrophobic(ligand_mol: Mol, target_mol: Mol) -> np.ndarray:
    ligand_indice = get_hydrophobic_atom(ligand_mol)
    target_indice = get_hydrophobic_atom(target_mol)
    return np.outer(ligand_indice, target_indice)


VDWRADII = {
    6: 1.90,
    7: 1.8,
    8: 1.7,
    16: 2.0,
    15: 2.1,
    9: 1.5,
    17: 1.8,
    35: 2.0,
    53: 2.2,
    30: 1.2,
    25: 1.2,
    26: 1.2,
    27: 1.2,
    12: 1.2,
    28: 1.2,
    20: 1.2,
    29: 1.2,
}
def get_vdw_radius(atom: Atom) -> float:
    atomic_num = atom.GetAtomicNum()
    if VDWRADII.get(atomic_num):
        return VDWRADII[atomic_num]
    return Chem.GetPeriodicTable().GetRvdw(atomic_num)



def pdb_to_graph(pdb):
    """ used in MoleculeDataset() class
    Converts rdkit mol objects to graph data object in pytorch geometric
    NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr """

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    pdbA = pdb_path + pdb + '_A.interface'
    pdbB = pdb_path + pdb + '_B.interface'

    mola = Chem.MolFromPDBFile(pdbA, sanitize=False)
    molb = Chem.MolFromPDBFile(pdbB, sanitize=False)

    hbond_AA = get_A_hbond(mola, mola)
    metal_AA = get_A_metal_complexes(mola, mola)
    hydrophobic_AA = get_A_hydrophobic(mola, mola)

    hbond_BB = get_A_hbond(molb, molb)
    metal_BB = get_A_metal_complexes(molb, molb)
    hydrophobic_BB = get_A_hydrophobic(molb, molb)

    hbond_AB = get_A_hbond(mola, molb)
    metal_AB = get_A_metal_complexes(mola, molb)
    hydrophobic_AB = get_A_hydrophobic(mola, molb)
    
    hbond_BA = get_A_hbond(molb, mola)
    metal_BA = get_A_metal_complexes(molb, mola)
    hydrophobic_BA = get_A_hydrophobic(molb, mola)

    edge_index_hbond = np.concatenate([np.concatenate([hbond_AA, hbond_AB], axis=1), np.concatenate([hbond_BA, hbond_BB], axis=1)], axis=0) # (N+M, N+M)
    edge_index_metal = np.concatenate([np.concatenate([metal_AA, metal_AB], axis=1), np.concatenate([metal_BA, metal_BB], axis=1)], axis=0) # (N+M, N+M)
    edge_index_hydrophobic = np.concatenate([np.concatenate([hydrophobic_AA, hydrophobic_AB], axis=1), np.concatenate([hydrophobic_BA, hydrophobic_BB], axis=1)], axis=0) # (N+M, N+M)

    np.fill_diagonal(edge_index_hbond, 0)
    np.fill_diagonal(edge_index_metal, 0)
    np.fill_diagonal(edge_index_hydrophobic, 0)


    edge_index_hbond = np.array(dense_to_sparse(torch.tensor(edge_index_hbond))[0])
    edge_index_metal = np.array(dense_to_sparse(torch.tensor(edge_index_metal))[0])
    edge_index_hydrophobic = np.array(dense_to_sparse(torch.tensor(edge_index_hydrophobic))[0])
    
    mola_vdw_radii = np.array(
        [get_vdw_radius(atom) for atom in mola.GetAtoms()]
    )
    molb_vdw_radii = np.array(
        [get_vdw_radius(atom) for atom in molb.GetAtoms()]
    )

    ligand_vdw_radii = np.concatenate([mola_vdw_radii, molb_vdw_radii], axis=0)
    target_vdw_radii = ligand_vdw_radii
    


    mola_and_non_metal = np.array(
            [
                1 if atom.GetSymbol() not in METALS else 0
                for atom in mola.GetAtoms()
            ]
        )
    molb_and_non_metal = np.array(
            [
                1 if atom.GetSymbol() not in METALS else 0
                for atom in molb.GetAtoms()
            ]
        )
    
    ligand_non_metal = np.concatenate([mola_and_non_metal, molb_and_non_metal], axis=0)
    target_non_metal = ligand_non_metal

    return edge_index_hbond, edge_index_metal, edge_index_hydrophobic, ligand_vdw_radii, target_vdw_radii, ligand_non_metal, target_non_metal



all_ppis = '/home/zqzhangzitong/project/PPIMI_HG/Data-2p2i-real/PPI_name.csv'
# ligands = pd.read_csv(all_ligands)['compound_id'].tolist()
ppis = pd.read_csv(all_ppis)['PPI'].tolist()

pdb_path = '/home/zqzhangzitong/project/PPIMI_HG/Data-2p2i-real/pdb_interface/5A/'
save_path = '/home/zqzhangzitong/project/PPIMI_HG/Data-2p2i-real/graphs/protein/5A/'

Edge_index_hbond=dict()
Edge_index_metal=dict()
Edge_index_hydrophobic=dict()
Ligand_vdw_radii=dict()
Target_vdw_radii=dict()
Ligand_non_metal = dict()
Target_non_metal = dict()

PPI_Engergy = dict()

for i in tqdm(range(len(ppis))):

    edge_index_hbond, edge_index_metal, edge_index_hydrophobic, ligand_vdw_radii, target_vdw_radii, ligand_non_metal, target_non_metal = pdb_to_graph(ppis[i])

    # Features[ppis[i]] = x
    # Edge_index[ppis[i]] = edge_index
    # Edge_attr[ppis[i]] = edge_attr

    Engergy = dict()
    Engergy['edge_index_hbond'] = edge_index_hbond
    Engergy['edge_index_metal'] = edge_index_metal
    Engergy['edge_index_hydrophobic'] = edge_index_hydrophobic
    Engergy['ligand_vdw_radii'] = ligand_vdw_radii
    Engergy['target_vdw_radii'] = target_vdw_radii
    Engergy['ligand_non_metal'] = ligand_non_metal
    Engergy['target_non_metal'] = target_non_metal
    PPI_Engergy[ppis[i]] = Engergy


    with open(save_path + 'ppi_energy.npy', "wb") as file:
        pickle.dump(PPI_Engergy, file)

