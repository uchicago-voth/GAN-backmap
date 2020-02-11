import sys
import os
import torch
import config as c
#def set_params(pair):
#    
#    
#    print(pair[0] + " set to " + pair[1])
#    if pair[0] == "BATCH_SIZE":
#        c.BATCH_SIZE = int(pair[1])
#    elif pair[0] == "NUM_EPOCHS":
#        c.NUM_EPOCHS = int(pair[1])
#    elif pair[0] == "PACKING_FACTOR":
#        c.PACKING_FACTOR = int(pair[1])
#    elif pair[0] == "GLEARNING_RATE":
#        c.GLEARNING_RATE = float(pair[1])
#    elif pair[0] == "DLEARNING_RATE":
#        c.DLEARNING_RATE = float(pair[1])
#    elif pair[0] == "Z_SIZE":
#        c.Z_SIZE = int(pair[1])
#    elif pair[0] == "D_TRAIN_RATE":
#        c.D_TRAIN_RATE = int(pair[1])
#    elif pair[0] == "TOTAL_SAMPLES":
#        c.TOTAL_SAMPLES = int(pair[1])
#    elif pair[0] == "G_WIDTH":
#        c.G_WIDTH = int(pair[1])
#    elif pair[0] == "D_WIDTH":
#        c.D_WIDTH = int(pair[1])
#    elif pair[0] == "GP_LAMBDA":
#        c.GP_LAMBDA = float(pair[1])
#    elif pair[0] == "NORM":
#        c.NORM = int(pair[1])
#    elif pair[0] == "AA_NUM_ATOMS":
#        c.AA_NUM_ATOMS = int(pair[1])
#    elif pair[0] == "CG_NUM_ATOMS":
#        c.CG_NUM_ATOMS = int(pair[1])
#    elif pair[0] == "CYCLE_LAMBDA":
#        c.CYCLE_LAMBDA = float(pair[1])
#    elif pair[0] == "NEIGHBORS":
#        c.NEIGHBORS = int(pair[1])
#    elif pair[0] == "cg_trajectory":
#        c.cg_trajectory = str(pair[1])
#    elif pair[0] == "aa_trajectory":
#        c.aa_trajectory = str(pair[1])
#    elif pair[0] == "cg_topology":
#        c.cg_topology = str(pair[1])
#    elif pair[0] == "aa_topology":
#        c.aa_topology = str(pair[1])
#    elif pair[0] == "G_DEPTH":
#        c.G_DEPTH = int(pair[1])
#    elif pair[0] == "model_output_freq":
#        c.model_output_freq = int(pair[1])
#    elif pair[0] == "D_DEPTH":
#        c.D_DEPTH = int(pair[1])
#    elif pair[0] == "device":
#        if pair[1] == "cpu":
#            c.device = torch.device('cpu')
#            c.ngpu = 0
#        elif pair[1] == "gpu":
#            c.device = torch.device('cuda:0')
#            c.ngpu = 1
#
#def read_input_file(name):
#    print("Reading input file")
#    input_file = open(name, "r")
#    for line in input_file:
#        set_params(line.split())
class config():
    def __init__(self):
        #Training Params
        self.BATCH_SIZE = 10
        self.NUM_EPOCHS = 1000
        self.PACKING_FACTOR = 1
        self.GLEARNING_RATE = 0.00008
        self.DLEARNING_RATE = 0.00005
        self.BETA1 = 0.5
        self.Z_SIZE = 20
        self.D_TRAIN_RATE = 10 
        self.G_WIDTH = 50
        self.D_WIDTH = 80
        self.CYCLE_LAMBDA = 10.0
        self.GP_LAMBDA = 10.0
        self.TOTAL_SAMPLES = 1000
        self.AA_NUM_ATOMS = 1
        self.CG_NUM_ATOMS = 1
        self.NUM_DIMS = 3
        self.NEIGHBORS = 1
        self.G_DEPTH = 4
        self.D_DEPTH = 8
        self.NORM = 1
        self.mode = 0


        #Dataset params
        self.device=torch.device('cpu')
        self.ngpu = 0
        #device=torch.device('cuda:0')
        #ngpu = 1
        
        self.model_output_freq = 100
        self.dataset_dir = 'data/'
        self.cg_trajectory = 'CG.lammpstrj'
        self.aa_trajectory = 'AA.lammpstrj'
        self.cg_topology = 'CG.pdb'
        self.aa_topology = 'AA.pdb'
        self.output_size = 100
        self.output_base = 'wgan_gp_outputs/'
        self.output_name = sys.argv[1]
        self.output_dir = self.output_base + self.output_name + '/'
        self.output_traj_name = self.output_name + '.lammpstrj'
        self.output_D_name = self.output_name + '_D.pth'
        self.output_G_name = self.output_name + '_G.pth'
        self.output_loss_name = self.output_name + "_losses.dat"
    
    
    
    
    def set_params(self, pair):
        print(pair[0] + " set to " + pair[1])
        if pair[0] == "BATCH_SIZE":
            self.BATCH_SIZE = int(pair[1])
        elif pair[0] == "NUM_EPOCHS":
            self.NUM_EPOCHS = int(pair[1])
        elif pair[0] == "PACKING_FACTOR":
            self.PACKING_FACTOR = int(pair[1])
        elif pair[0] == "GLEARNING_RATE":
            self.GLEARNING_RATE = float(pair[1])
        elif pair[0] == "DLEARNING_RATE":
            self.DLEARNING_RATE = float(pair[1])
        elif pair[0] == "Z_SIZE":
            self.Z_SIZE = int(pair[1])
        elif pair[0] == "D_TRAIN_RATE":
            self.D_TRAIN_RATE = int(pair[1])
        elif pair[0] == "TOTAL_SAMPLES":
            self.TOTAL_SAMPLES = int(pair[1])
        elif pair[0] == "G_WIDTH":
            self.G_WIDTH = int(pair[1])
        elif pair[0] == "D_WIDTH":
            self.D_WIDTH = int(pair[1])
        elif pair[0] == "GP_LAMBDA":
            self.GP_LAMBDA = float(pair[1])
        elif pair[0] == "NORM":
            self.NORM = int(pair[1])
        elif pair[0] == "AA_NUM_ATOMS":
            self.AA_NUM_ATOMS = int(pair[1])
        elif pair[0] == "CG_NUM_ATOMS":
            self.CG_NUM_ATOMS = int(pair[1])
        elif pair[0] == "CYCLE_LAMBDA":
            self.CYCLE_LAMBDA = float(pair[1])
        elif pair[0] == "NEIGHBORS":
            self.NEIGHBORS = int(pair[1])
        elif pair[0] == "cg_trajectory":
            self.cg_trajectory = str(pair[1])
        elif pair[0] == "aa_trajectory":
            self.aa_trajectory = str(pair[1])
        elif pair[0] == "cg_topology":
            self.cg_topology = str(pair[1])
        elif pair[0] == "aa_topology":
            self.aa_topology = str(pair[1])
        elif pair[0] == "G_DEPTH":
            self.G_DEPTH = int(pair[1])
        elif pair[0] == "model_output_freq":
            self.model_output_freq = int(pair[1])
        elif pair[0] == "output_size":
            self.output_size = int(pair[1])
        elif pair[0] == "D_DEPTH":
            self.D_DEPTH = int(pair[1])
        elif pair[0] == "device":
            if pair[1] == "cpu":
                self.device = torch.device('cpu')
                self.ngpu = 0
            elif pair[1] == "gpu":
                self.device = torch.device('cuda:0')
                self.ngpu = 1
        elif pair[0] == "mode":
            if pair[1] == "distance":
                self.mode = 0
            elif pair[1] == "internal":
                self.mode = 1


 
    def read_input_file(self, name):
        print("Reading input file")
        input_file = open(name, "r")
        for line in input_file:
            self.set_params(line.split())
