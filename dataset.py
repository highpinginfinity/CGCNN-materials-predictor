import csv
import json
import os
import torch
from torch.utils.data import Dataset
from pymatgen.core.structure import Structure
import numpy as np

class CIFDataset(Dataset):
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.id_prop_data = []
        
        with open(os.path.join(root_dir, 'id_prop.csv')) as f:
            reader = csv.reader(f)
            self.id_prop_data = [[row[0], float(row[1])] for row in reader]
            
        with open(os.path.join(root_dir, 'atom_init.json')) as f:
            self.ari = json.load(f)

        # Precompute Gaussian filter for bond distances
        self.filter = np.arange(dmin, radius + step, step)

    def __len__(self):
        return len(self.id_prop_data)

    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir, f"{cif_id}.cif"))
        
        # Get atom features
        atom_fea = np.vstack([self.ari[str(crystal[i].specie.number)] for i in range(len(crystal))])
        
        # Get neighbors and bond distances
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1])[:self.max_num_nbr] for nbrs in all_nbrs]
        
        nbr_fea_idx, nbr_fea = [], []
        for nbrs in all_nbrs:
            nbr_fea_idx.append([x[2] for x in nbrs] + [0] * (self.max_num_nbr - len(nbrs)))
            distances = [x[1] for x in nbrs] + [self.radius + 1.] * (self.max_num_nbr - len(nbrs))
            # Apply Gaussian expansion to distances
            expanded = np.exp(-(np.array(distances)[..., np.newaxis] - self.filter)**2 / 0.5)
            nbr_fea.append(expanded)
            
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(np.array(nbr_fea))
        nbr_fea_idx = torch.LongTensor(np.array(nbr_fea_idx))
        target = torch.Tensor([target])
        
        return atom_fea, nbr_fea, nbr_fea_idx, target, cif_id