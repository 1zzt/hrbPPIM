import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
import pickle
from tqdm import tqdm


def create_hetergraph(lig_x, lig_edge_index, lig_edge_attr, ppi_x, ppi_edge_attr, 
                          ppi_cov_edge_index, ppi_noncov_edge_index, ppi_pos,ppi_energy, label):

        data = HeteroData()  

        data.y = torch.tensor(label)
        

        data['ligand'].x = torch.tensor(lig_x, dtype=torch.long)

        data['ligand', 'l2l', 'ligand'].edge_index =  torch.tensor(lig_edge_index, dtype=torch.long)
        data['ligand', 'l2l', 'ligand'].edge_attr =  torch.tensor(lig_edge_attr, dtype=torch.long)

        ppi_all_edges = np.concatenate((ppi_cov_edge_index, ppi_noncov_edge_index), axis=1)
        data['protein'].x = torch.tensor(ppi_x, dtype=torch.float)
        data['protein'].pos = torch.tensor(ppi_pos, dtype=torch.float)
        data['protein', 'p2p_cov', 'protein'].edge_index = torch.tensor(ppi_cov_edge_index, dtype=torch.long)
        data['protein', 'p2p_cov', 'protein'].edge_attr = torch.tensor(ppi_edge_attr, dtype=torch.long)
        data['protein', 'p2p_noncov', 'protein'].edge_index = torch.tensor(ppi_noncov_edge_index, dtype=torch.long)
        data['protein', 'p2p_alledges', 'protein'].edge_index = torch.tensor(ppi_all_edges, dtype=torch.long)


        data['protein', 'hbond', 'protein'].edge_index = torch.tensor(ppi_energy['edge_index_hbond'], dtype=torch.long)
        data['protein', 'metal', 'protein'].edge_index = torch.tensor(ppi_energy['edge_index_metal'], dtype=torch.long)
        data['protein', 'hydrophobic', 'protein'].edge_index = torch.tensor(ppi_energy['edge_index_hydrophobic'], dtype=torch.long)
        data['protein'].ligand_vdw_radii = torch.tensor(ppi_energy['ligand_vdw_radii'], dtype=torch.float)
        data['protein'].target_vdw_radii = torch.tensor(ppi_energy['target_vdw_radii'], dtype=torch.float)
        data['protein'].ligand_non_metal = torch.tensor(ppi_energy['ligand_non_metal'], dtype=torch.float)
        data['protein'].target_non_metal = torch.tensor(ppi_energy['target_non_metal'], dtype=torch.float)

        return data


class PPIMIDataset(Dataset):
    def __init__(self, mode, fold):
        super(PPIMIDataset, self).__init__()

        datapath = f"/home/data/fold-{fold}-{mode}.csv"
        print('datapath\t', datapath) 


        df = pd.read_csv(datapath)
        self.ligands = df['SMILES'].tolist()
        self.ppis = df['PPI'].tolist()
        self.labels = df['label'].tolist()

        self.data_list = []
        self.process()
        

    

    def process(self):

        graph_path = '/home/data/graphs'

        lig_x = pickle.load(open(f'{graph_path}/ligand/ligand_x.npy', 'rb'))
        lig_edge_index = pickle.load(open(f'{graph_path}/ligand/edge_index.npy', 'rb'))
        lig_edge_attr = pickle.load(open(f'{graph_path}/ligand/edge_attr.npy', 'rb'))

        ppi_x = pickle.load(open(f'{graph_path}/protein/covx.npy', 'rb'))
        ppi_edge_attr = pickle.load(open(f'{graph_path}/protein/edge_attr.npy', 'rb'))
        ppi_cov_edge_index = pickle.load(open(f'{graph_path}/protein/edge_index.npy', 'rb'))
        ppi_noncov_edge_index = pickle.load(open(f'{graph_path}/protein/noncov_edge_index.npy', 'rb')) 
        ppi_pos = pickle.load(open(f'{graph_path}/protein/pos.npy', 'rb'))
        ppi_energy = pickle.load(open(f'{graph_path}/protein/ppi_energy.npy', 'rb'))



        for i in tqdm(range(len(self.labels))):
            smiles = self.ligands[i]
            ppi_name = self.ppis[i]
            label = self.labels[i]


            data = create_hetergraph(lig_x[smiles], lig_edge_index[smiles], lig_edge_attr[smiles], ppi_x[ppi_name], ppi_edge_attr[ppi_name], 
                          ppi_cov_edge_index[ppi_name], ppi_noncov_edge_index[ppi_name], ppi_pos[ppi_name], ppi_energy[ppi_name], label)

            self.data_list.append(data)


        print(f"{len(self.data_list)} graphs have been created!")


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]     