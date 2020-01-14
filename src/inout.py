import sys
import os
import torch
import config as c
def set_params(pair):
    
    
    print(pair[0] + " set to " + pair[1])
    if pair[0] == "BATCH_SIZE":
        c.BATCH_SIZE = int(pair[1])
    elif pair[0] == "NUM_EPOCHS":
        c.NUM_EPOCHS = int(pair[1])
    elif pair[0] == "PACKING_FACTOR":
        c.PACKING_FACTOR = int(pair[1])
    elif pair[0] == "GLEARNING_RATE":
        c.GLEARNING_RATE = float(pair[1])
    elif pair[0] == "DLEARNING_RATE":
        c.DLEARNING_RATE = float(pair[1])
    elif pair[0] == "Z_SIZE":
        c.Z_SIZE = int(pair[1])
    elif pair[0] == "D_TRAIN_RATE":
        c.D_TRAIN_RATE = int(pair[1])
    elif pair[0] == "TOTAL_SAMPLES":
        c.TOTAL_SAMPLES = int(pair[1])
    elif pair[0] == "G_WIDTH":
        c.G_WIDTH = int(pair[1])
    elif pair[0] == "D_WIDTH":
        c.D_WIDTH = int(pair[1])
    elif pair[0] == "GP_LAMBDA":
        c.GP_LAMBDA = float(pair[1])
    elif pair[0] == "NORM":
        c.NORM = int(pair[1])
    elif pair[0] == "AA_NUM_ATOMS":
        c.AA_NUM_ATOMS = int(pair[1])
    elif pair[0] == "CG_NUM_ATOMS":
        c.CG_NUM_ATOMS = int(pair[1])
    elif pair[0] == "CYCLE_LAMBDA":
        c.CYCLE_LAMBDA = float(pair[1])
    elif pair[0] == "NEIGHBORS":
        c.NEIGHBORS = int(pair[1])
    elif pair[0] == "cg_trajectory":
        c.cg_trajectory = str(pair[1])
    elif pair[0] == "aa_trajectory":
        c.aa_trajectory = str(pair[1])
    elif pair[0] == "cg_topology":
        c.cg_topology = str(pair[1])
    elif pair[0] == "aa_topology":
        c.aa_topology = str(pair[1])
    elif pair[0] == "G_DEPTH":
        c.G_DEPTH = int(pair[1])
    elif pair[0] == "model_output_freq":
        c.model_output_freq = int(pair[1])
    elif pair[0] == "D_DEPTH":
        c.D_DEPTH = int(pair[1])
    elif pair[0] == "device":
        if pair[1] == "cpu":
            c.device = torch.device('cpu')
            c.ngpu = 0
        elif pair[1] == "gpu":
            c.device = torch.device('cuda:0')
            c.ngpu = 1

def read_input_file(name):
    print("Reading input file")
    input_file = open(name, "r")
    for line in input_file:
        set_params(line.split())

