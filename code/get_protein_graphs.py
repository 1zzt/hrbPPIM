
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from scipy.spatial import distance




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



def get_edgeindex(mol):

    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 3  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        # edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edges_list.append((i, j))
            edges_list.append((j, i))

        edge_index = np.array(edges_list).T
    
    return edge_index


def compute_graph(posA, posB, cutoff = 5):

    pairwise_dists = distance.cdist(posA, posB)

    src_list = []
    dst_list = []

    for i in range(pairwise_dists.shape[0]):

        dst = list(np.where(pairwise_dists[i, :] < cutoff)[0])
        src = [i] * len(dst)
        dst_list.extend(dst)
        src_list.extend(src)
    
    return src_list, dst_list

def get_mol_features(mol):

    atom_features = [(atom.GetIdx(), get_atom_features(atom)) for atom in mol.GetAtoms()]
    atom_features.sort() # to make sure that the feature matrix is aligned according to the idx of the atom
    _, atom_features = zip(*atom_features)
    atom_features = torch.stack(atom_features)

    return atom_features



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

    
    x_A = get_mol_features(mola)
    x_B = get_mol_features(molb)

    pos_A = mola.GetConformers()[0].GetPositions()
    pos_B = molb.GetConformers()[0].GetPositions()

    num_atomA = mola.GetNumAtoms()
    src_list, dst_list = compute_graph(pos_A, pos_B, cutoff = 5)
    dst_list = [x + num_atomA for x in dst_list]

    noncov_edge_index = np.array([src_list+dst_list, dst_list+src_list])

    x = np.concatenate((x_A, x_B), axis=0)
    pos = np.concatenate((pos_A, pos_B), axis=0)

    # bonds
    cov_edge_index_A = get_edgeindex(mola)
    cov_edge_index_B = get_edgeindex(molb)

    cov_edge_attr_A = get_bond_feature(mola, cov_edge_index_A)
    cov_edge_attr_B = get_bond_feature(molb, cov_edge_index_B)


    cov_edge_index_B = [x + num_atomA for x in cov_edge_index_B]

    cov_edge_index = np.concatenate((cov_edge_index_A, cov_edge_index_B), axis=1)
    cov_edge_attr = np.concatenate((cov_edge_attr_A, cov_edge_attr_B), axis=0)


    
    return x, pos, cov_edge_index, cov_edge_attr, noncov_edge_index



all_ppis = '/home/PPI_name.csv'
ppis = pd.read_csv(all_ppis)['PPI'].tolist()

pdb_path = '/home/pdb_interface/'
save_path = '/home/graphs/protein/'


Features=dict()
Poss=dict()
Cov_edge_index=dict()
Cov_edge_attr=dict()
NonCov_edge_index=dict()


for i in tqdm(range(len(ppis))):

    x, pos, cov_edge_index, cov_edge_attr, noncov_edge_index = pdb_to_graph(ppis[i])
    Features[ppis[i]] = x
    Poss[ppis[i]] = pos
    Cov_edge_index[ppis[i]] = cov_edge_index
    Cov_edge_attr[ppis[i]] = cov_edge_attr
    NonCov_edge_index[ppis[i]] = noncov_edge_index




with open(save_path + 'covx.npy', "wb") as file:
    pickle.dump(Features, file)
with open(save_path + 'pos.npy', "wb") as file:
    pickle.dump(Poss, file)
with open(save_path + 'edge_index.npy', "wb") as file:
    pickle.dump(Cov_edge_index, file)
with open(save_path + 'edge_attr.npy', "wb") as file:
    pickle.dump(Cov_edge_attr, file)
with open(save_path + 'noncov_edge_index.npy', "wb") as file:
    pickle.dump(NonCov_edge_index, file)