# hrbPPIM
An end-to-end deep learning approach that introduces three key inductive biases of the PPI modulatory system: geometric invariance, hierarchical representation, and interaction-aware modeling. By encoding PPI complexes with rigid-body invariant embeddings, dynamically clustering atoms into chemically meaningful substructures, and injecting physicochemical priors into attention-based cross-entity modeling, hrbPPIM constrains the hypothesis space to biologically plausible interaction mechanisms.

![image](https://github.com/1zzt/hrbPPIM/raw/main/Overview.png)

##  Data preparation:
 - Experimentally validated PPI-modulator interaction pairs in the benchmark dataset: https://2p2idb.marseille.inserm.fr/index.html;
 - Details of the 1755 processed interaction entries are available at /Data/pos_2p2i.csv;
 - DLiP-KF dataset: https://github.com/1zzt/KFPPIMI;
 - Decoy compounds in virtual screening datasets: https://zinc15.docking.org/;
 - SARS-CoV-2 assays data: https://opendata.ncats.nih.gov/covid19/assays;
 - Approved drugs in DrugBank: https://go.drugbank.com/releases/5-1-13/downloads/approved-drug-links;
 - protein structures can be downloaded from: https://www.rcsb.org/.

## Pepline
### 1. Graph construction
1\) Run `get_comp_graphs.py` to extract protein complex interfaces. Then run `de.py` to construct graph representations and save them as .npy files.
2\) Run `get_protein_graphs.py` to construct compound graph representations and save them as .npy files.
### 2. PPI energy calculation
Run `get_ppi_energy.py` to extract protein interaction energy. The input consists of two protein interfaces, and the output is saved in an `.npy` file.
### 3. Train
Run `main.py` to train and evaluate the model.

Replace the input and output paths with your own.
