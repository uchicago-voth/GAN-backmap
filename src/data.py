import torch
import torch.nn as nn
import torch.utils.data as data_utils
import pandas as pd
import numpy as np
import mdtraj as md
import fileinput
from abc import ABC


#Normalization functions
#def norm(array, minimum, maximum):
#    return np.multiply(np.subtract(np.divide(np.subtract(array, minimum),np.subtract(maximum, minimum)), 0.5),2.0)
#def unnorm(array, minimum, maximum):
#    return np.add(np.multiply(array/2+0.5, np.subtract(maximum, minimum)), minimum)

#Abstract dataset constructor class, not to be created
class _Dataset_Constructor(ABC):
    def __init__(self, param):
        self.BATCH_SIZE = param.BATCH_SIZE
        self.TOTAL_SAMPLES = param.TOTAL_SAMPLES
        self.AA_NUM_ATOMS = self.aa_trj.shape[1]
        self.CG_NUM_ATOMS = self.cg_trj.shape[1]
        self.min_1 = self.aa_trj[:,:,0].min()
        self.min_2 = self.aa_trj[:,:,1].min()
        self.min_3 = self.aa_trj[:,:,2].min()
        self.max_1 = self.aa_trj[:,:,0].max()
        self.max_2 = self.aa_trj[:,:,1].max()
        self.max_3 = self.aa_trj[:,:,2].max()


    def norm(self, array):
        array[:,:,0] = np.multiply(np.subtract(np.divide(np.subtract(array[:,:,0], self.min_1),np.subtract(self.max_1, self.min_1)), 0.5),2.0)
        array[:,:,1] = np.multiply(np.subtract(np.divide(np.subtract(array[:,:,1], self.min_2),np.subtract(self.max_2, self.min_2)), 0.5),2.0)
        array[:,:,2] = np.multiply(np.subtract(np.divide(np.subtract(array[:,:,2], self.min_3),np.subtract(self.max_3, self.min_3)), 0.5),2.0)
        return array
        #return np.multiply(np.subtract(np.divide(np.subtract(array, self.min_data),np.subtract(self.max_data, self.min_data)), 0.5),2.0)
    
    def unnorm(self, array):
        array[:,:,0] = np.add(np.multiply(array[:,:,0]/2+0.5, np.subtract(self.max_1, self.min_1)), self.min_1)
        array[:,:,1] = np.add(np.multiply(array[:,:,1]/2+0.5, np.subtract(self.max_2, self.min_2)), self.min_2)
        array[:,:,2] = np.add(np.multiply(array[:,:,2]/2+0.5, np.subtract(self.max_3, self.min_3)), self.min_3)
        return array
        #return np.add(np.multiply(array/2+0.5, np.subtract(self.max_data, self.min_data)), self.min_data)


    def construct_data_loader(self, trj, NUM_ATOMS):
        coordinates = trj
        #coordinates = trj.atom_slice(range(NUM_ATOMS), inplace=True).xyz[0:self.TOTAL_SAMPLES]
        data_normalized = self.norm(coordinates)
        samples = torch.from_numpy(data_normalized)
        data = data_utils.TensorDataset(samples)
        data_loader = data_utils.DataLoader(data, batch_size = self.BATCH_SIZE, shuffle=True, drop_last=True)
        return data_loader





#Cartesian dataset. Usable for direct cartesian coordinates, or distance matrices
class Cartesian_Dataset_Constructor(_Dataset_Constructor):
    def __init__(self, param):
        self.dataset_dir = param.dataset_dir
        self.aa_trj = md.load(param.dataset_dir + param.aa_trajectory, top=param.dataset_dir + param.aa_topology)
        self.cg_trj = md.load(param.dataset_dir + param.cg_trajectory, top=param.dataset_dir + param.cg_topology) 
        self.table, self.bonds = self.aa_trj.top.to_dataframe()
        self.aa_trj = self.aa_trj.xyz[0:param.TOTAL_SAMPLES]
        self.cg_trj = self.cg_trj.xyz[0:param.TOTAL_SAMPLES]
        super().__init__(param)




    def res_atom_counts(self):
        atom_counts = []
        atom=0
        for i in range(self.CG_NUM_ATOMS):
            subtable = self.table.loc[self.table['resSeq'] == i+1]
            for i in subtable.itertuples():
                atom+=1
            atom_counts.append(atom)
        return atom_counts
        
    def input_atom_counts(self, filename):
        f = open(filename, "r")
        atom_counts = []
        for i in f.readlines():
            atom_counts.append(int(i))
        return atom_counts

    

    def construct_forward_matrix(self, atom_counts, device): 
        massdict = {
                'H': 1.008,
                'C': 12.011,
                'N': 14.007,
                'O': 15.999,
                'P': 30.974,
                'S': 32.065
         }
        
        if len(atom_counts) != self.CG_NUM_ATOMS:
            raise SystemExit('Error: CG correspondance file is inconsistent with number of CG sites')
        self.table["element"].replace(massdict, inplace=True)
        M = np.zeros((self.AA_NUM_ATOMS, self.CG_NUM_ATOMS))
        prev = 0
        atom=0
        for i in range(len(atom_counts)):
            subtable = self.table[prev:atom_counts[i]]
            total_mass = subtable["element"].sum()
            for j in subtable.itertuples():
                mass_frac = j.element/total_mass
                M[atom, i] = mass_frac
                atom+=1
            prev=atom_counts[i]
        G_f = torch.from_numpy(M).type(torch.float).to(device)
        return G_f







#Internal coord dataset constructor. Usable with .bad files (bonds angles and dihedrals). These are created with xyz_zmat_util.py
class Internal_Dataset_Constructor(_Dataset_Constructor):
    def __init__(self, param):
        self.aa_trj = np.genfromtxt(param.dataset_dir + param.aa_trajectory, delimiter=',').swapaxes(0,1).reshape(self.TOTAL_SAMPLES, -1, 3)
        self.cg_trj = np.genfromtxt(param.dataset_dir + param.cg_trajectory, delimiter=',').swapaxes(0,1).reshape(self.TOTAL_SAMPLES, -1, 3)
        super().__init__(BATCH_SIZE, TOTAL_SAMPLES)




#Generate forward mapping matrix
#def forward_matrix(AA_NUM_ATOMS, CG_NUM_ATOMS, aa_trj, device):
#     table, bonds = aa_trj.top.to_dataframe()
#     massdict = {
#             'H': 1.008,
#             'C': 12.011,
#             'N': 14.007,
#             'O': 15.999,
#             'P': 30.974,
#             'S': 32.065
#     }
#     atom_counts = []
#     table["element"].replace(massdict, inplace=True)
#     M = np.zeros((AA_NUM_ATOMS , CG_NUM_ATOMS ))
#     atom = 0
#     ones_count = 0
#     for i in range(CG_NUM_ATOMS):
#         subtable = table.loc[table['resSeq'] == i+1]
#         total_mass = subtable["element"].sum()
#         count = 0
#         for j in subtable.itertuples():
#             mass_frac = j.element/total_mass
#             M[atom, i] = mass_frac
#             atom+=1
#         atom_counts.append(atom)
#     G_f = torch.from_numpy(M).type(torch.float).to(device)
#     return G_f, atom_counts



#Make the dataset

#def construct_dataset(BATCH_SIZE, TOTAL_SAMPLES, dataset_dir, cg_trajectory, aa_trajectory, cg_topology, aa_topology):
#    
#    cg_trj = md.load(dataset_dir + cg_trajectory, top=dataset_dir + cg_topology)
#    CG_NUM_ATOMS = cg_trj.xyz.shape[1]
#    cg_coordinates = cg_trj.atom_slice(range(CG_NUM_ATOMS), inplace=True).xyz[0:TOTAL_SAMPLES]
#    
#    aa_trj = md.load(dataset_dir + aa_trajectory, top=dataset_dir + aa_topology)
#    AA_NUM_ATOMS = aa_trj.xyz.shape[1]
#    aa_coordinates = aa_trj.atom_slice(range(AA_NUM_ATOMS), inplace=True).xyz[0:TOTAL_SAMPLES]
#          
#    min_data = aa_coordinates.min()
#    max_data = aa_coordinates.max()
#    cg_data_normalized = norm(cg_coordinates, min_data, max_data)
#    aa_data_normalized = norm(aa_coordinates, min_data, max_data)
#    cg_samples = torch.from_numpy(cg_data_normalized)
#    aa_samples = torch.from_numpy(aa_data_normalized)
#    
#    cg_data = data_utils.TensorDataset(cg_samples)
#    cg_data_loader = data_utils.DataLoader(cg_data, batch_size=BATCH_SIZE, shuffle=True)
#     
#    aa_data = data_utils.TensorDataset(aa_samples)
#    aa_data_loader = data_utils.DataLoader(aa_data, batch_size=BATCH_SIZE, shuffle=True)
#
#    return aa_data_loader, cg_data_loader, aa_trj, cg_samples, CG_NUM_ATOMS, AA_NUM_ATOMS, min_data, max_data



