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

from rdkit.Chem.rdmolops import GetAdjacencyMatrix


load_file = 'all_smiles_file.csv'
ligand_smiles = pd.read_csv(load_file)['smiles'].tolist()

save_path = '/home/graphs/ligand/'


ALLOWABLE_BOND_FEATURES = {
    'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
    'conjugated': ['T/F'],
    'stereo': ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY']
}


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_atom_features(atom,
                explicit_H=True,
                use_chirality=False):

    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
        ]) + [atom.GetDegree()/10, atom.GetImplicitValence(), 
                atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, 
                Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()] + [atom.GetMass() * 0.01] + [int(atom.GetChiralTag())]

    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]
    
    

    results = np.array(results).astype(np.float32)
    return torch.from_numpy(results)


def get_bond_feature(mol, edge_index):
    bond_features = []    
    for i in range(edge_index.shape[1]):
        a1 = int(edge_index[0][i])
        a2 = int(edge_index[1][i])
        if a1 != a2:
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond is not None:  
                bond_features.append(np.array(
                        one_of_k_encoding_unk(str(bond.GetBondType()), ALLOWABLE_BOND_FEATURES['bond_type']) +
                        [bond.GetIsConjugated()] +
                        one_of_k_encoding_unk(str(bond.GetStereo()), ALLOWABLE_BOND_FEATURES['stereo'])
                    ))

    return torch.from_numpy(np.array(bond_features)).float()



def mol2graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    rows, cols = np.nonzero(GetAdjacencyMatrix(mol))
    rows = torch.tensor(rows, dtype=torch.long)
    cols = torch.tensor(cols, dtype=torch.long)
    edge_index = torch.stack([rows, cols], dim = 0)
    
    atom_features = [(atom.GetIdx(), get_atom_features(atom)) for atom in mol.GetAtoms()]
    atom_features.sort() # to make sure that the feature matrix is aligned according to the idx of the atom
    _, atom_features = zip(*atom_features)
    atom_features = torch.stack(atom_features)
    
    bond_features = get_bond_feature(mol, edge_index)

    return atom_features, edge_index, bond_features


def smiles2graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print(smiles)

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atom_features_list.append(atom_feature)
    x = np.array(atom_features_list)
    # pos = torch.FloatTensor(mol.GetConformers()[0].GetPositions())

    # bonds
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 3  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = np.array(edges_list).T
        edge_attr = np.array(edge_features_list)

    
    return x, edge_index, edge_attr



Features=dict()
Edge_index=dict()
Edge_attr=dict()

for i in tqdm(range(len(ligand_smiles))):


    x, edge_index, edge_attr = smiles2graph(ligand_smiles[i])

    Features[ligand_smiles[i]] = x
    Edge_index[ligand_smiles[i]] = edge_index
    Edge_attr[ligand_smiles[i]] = edge_attr

with open(save_path + 'graphmvp_x.npy', "wb") as file:
    pickle.dump(Features, file)
with open(save_path + 'edge_index.npy', "wb") as file:
    pickle.dump(Edge_index, file)
with open(save_path + 'graphmvp_edge_attr.npy', "wb") as file:
    pickle.dump(Edge_attr, file)