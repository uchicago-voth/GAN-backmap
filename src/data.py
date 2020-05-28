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
        self.AA_NUM_ATOMS = param.AA_NUM_ATOMS 
        self.CG_NUM_ATOMS = param.CG_NUM_ATOMS 
        self.aa_minmax = np.zeros((2,3))
        self.cg_minmax = np.zeros((2,3))
        for i in range(3):
            self.aa_minmax[0,i] = self.aa_trj[:,:,i].min()
            self.aa_minmax[1,i] = self.aa_trj[:,:,i].max()
            self.cg_minmax[0,i] = self.cg_trj[:,:,i].min()
            self.cg_minmax[1,i] = self.cg_trj[:,:,i].max()
        print(self.aa_minmax)
        print(self.cg_minmax)



    def norm(self, array, minmax):
        array2 = array
        array2[:,:,0] = np.multiply(np.subtract(np.divide(np.subtract(array[:,:,0], minmax[0,0]),np.subtract(minmax[1,0], minmax[0,0])), 0.5),2.0)
        array2[:,:,1] = np.multiply(np.subtract(np.divide(np.subtract(array[:,:,1], minmax[0,1]),np.subtract(minmax[1,1], minmax[0,1])), 0.5),2.0)
        array2[:,:,2] = np.multiply(np.subtract(np.divide(np.subtract(array[:,:,2], minmax[0,2]),np.subtract(minmax[1,2], minmax[0,2])), 0.5),2.0)
        return array2
    
    def unnorm(self, array, minmax):
        array2 = array
        array2[:,:,0] = np.add(np.multiply(array[:,:,0]/2+0.5, np.subtract(minmax[1,0], minmax[0,0])), minmax[0,0])
        array2[:,:,1] = np.add(np.multiply(array[:,:,1]/2+0.5, np.subtract(minmax[1,1], minmax[0,1])), minmax[0,1])
        array2[:,:,2] = np.add(np.multiply(array[:,:,2]/2+0.5, np.subtract(minmax[1,2], minmax[0,2])), minmax[0,2])
        return array2


    def construct_data_loader(self, trj, NUM_ATOMS, minmax):
        coordinates = trj.copy()
        #coordinates = trj.atom_slice(range(NUM_ATOMS), inplace=True).xyz[0:self.TOTAL_SAMPLES]
        data_normalized = self.norm(coordinates, minmax)
        samples = torch.from_numpy(data_normalized)
        data = data_utils.TensorDataset(samples)
        data_loader = data_utils.DataLoader(data, batch_size = self.BATCH_SIZE, shuffle=True, drop_last=True)
        return data_loader





#Cartesian dataset. Usable for direct cartesian coordinates, or distance matrices
class Cartesian_Dataset_Constructor(_Dataset_Constructor):
    def __init__(self, param):
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
        self.aa_trj = np.genfromtxt(param.dataset_dir + param.aa_trajectory, delimiter=',')
        self.aa_trj = self.aa_trj.reshape(-1, param.AA_NUM_ATOMS, 3)
        self.aa_trj = self.aa_trj[0:param.TOTAL_SAMPLES, :, :]
     
        
        self.cg_trj = np.genfromtxt(param.dataset_dir + param.cg_trajectory, delimiter=',')
        self.cg_trj = self.cg_trj.reshape(-1, param.CG_NUM_ATOMS, 3)
        self.cg_trj = self.cg_trj[0:param.TOTAL_SAMPLES, :, :]
       
        super().__init__(param)


class Internal_Fragment_Dataset_Constructor(_Dataset_Constructor):
    def __init__(self, param):
        self.aa_trj = np.genfromtxt(param.dataset_dir + param.aa_trajectory, delimiter=',')
        self.aa_trj = self.aa_trj.reshape(-1, param.AA_NUM_ATOMS, 3)
        self.aa_trj = self.aa_trj[0:param.TOTAL_SAMPLES, :, :]

        
        self.cg_trj = md.load(param.dataset_dir + param.cg_trajectory, top=param.dataset_dir + param.cg_topology) 
        self.cg_trj = self.cg_trj.xyz[0:param.TOTAL_SAMPLES]
        super().__init__(param)




