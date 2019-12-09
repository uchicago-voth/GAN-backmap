import torch
import torch.nn as nn
import torch.utils.data as data_utils
import pandas as pd
import numpy as np
import mdtraj as md


#Normalization functions
def norm(array, minimum, maximum):
    return np.multiply(np.subtract(np.divide(np.subtract(array, minimum),np.subtract(maximum, minimum)), 0.5),2.0)
def unnorm(array, minimum, maximum):
    return np.add(np.multiply(array/2+0.5, np.subtract(maximum, minimum)), minimum)


#Generate forward mapping matrix
def forward_matrix(AA_NUM_ATOMS, CG_NUM_ATOMS, aa_trj, device):
     table, bonds = aa_trj.top.to_dataframe()
     massdict = {
             'H': 1.008,
             'C': 12.011,
             'N': 14.007,
             'O': 15.999,
             'P': 30.974,
             'S': 32.065
     }
     atom_counts = []
     table["element"].replace(massdict, inplace=True)
     M = np.zeros((AA_NUM_ATOMS , CG_NUM_ATOMS ))
     atom = 0
     ones_count = 0
     for i in range(CG_NUM_ATOMS):
         subtable = table.loc[table['resSeq'] == i+1]
         total_mass = subtable["element"].sum()
         count = 0
         for j in subtable.itertuples():
             mass_frac = j.element/total_mass
             M[atom, i] = mass_frac
             atom+=1
         atom_counts.append(atom)
     G_f = torch.from_numpy(M).type(torch.float).to(device)
     return G_f, atom_counts



#Make the dataset

def construct_dataset(BATCH_SIZE, TOTAL_SAMPLES, dataset_dir, cg_trajectory, aa_trajectory, cg_topology, aa_topology):
    
    cg_trj = md.load(dataset_dir + cg_trajectory, top=dataset_dir + cg_topology)
    CG_NUM_ATOMS = cg_trj.xyz.shape[1]
    cg_coordinates = cg_trj.atom_slice(range(CG_NUM_ATOMS), inplace=True).xyz[0:TOTAL_SAMPLES]
    
    aa_trj = md.load(dataset_dir + aa_trajectory, top=dataset_dir + aa_topology)
    AA_NUM_ATOMS = aa_trj.xyz.shape[1]
    aa_coordinates = aa_trj.atom_slice(range(AA_NUM_ATOMS), inplace=True).xyz[0:TOTAL_SAMPLES]
          
    min_data = aa_coordinates.min()
    max_data = aa_coordinates.max()
    cg_data_normalized = norm(cg_coordinates, min_data, max_data)
    aa_data_normalized = norm(aa_coordinates, min_data, max_data)
    cg_samples = torch.from_numpy(cg_data_normalized)
    aa_samples = torch.from_numpy(aa_data_normalized)
    
    cg_data = data_utils.TensorDataset(cg_samples)
    cg_data_loader = data_utils.DataLoader(cg_data, batch_size=BATCH_SIZE, shuffle=True)
     
    aa_data = data_utils.TensorDataset(aa_samples)
    aa_data_loader = data_utils.DataLoader(aa_data, batch_size=BATCH_SIZE, shuffle=True)

    return aa_data_loader, cg_data_loader, aa_trj, cg_samples, CG_NUM_ATOMS, AA_NUM_ATOMS, min_data, max_data



