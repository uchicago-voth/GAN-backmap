import sys
import torch
#List of constants and defaults


global BATCH_SIZE, NUM_EPOCHS, PACKING_FACTOR, GLEARNING_RATE, DLEARNING_RATE, Z_SIZE, D_TRAIN_RATE, G_WIDTH, D_WIDTH, GP_LAMBDA, CYCLE_LAMBDA, TOTAL_SAMPLES, AA_NUM_ATOMS, CG_NUM_ATOMS, NEIGHBORS, cg_trajectory, aa_trajectory, cg_topology, aa_topology, device, ngpu

#Training Params
BATCH_SIZE = 10
NUM_EPOCHS = 1000
PACKING_FACTOR = 1
GLEARNING_RATE = 0.00008
DLEARNING_RATE = 0.00005
BETA1 = 0.5
Z_SIZE = 20
D_TRAIN_RATE = 10 
G_WIDTH = 50
D_WIDTH = 80
CYCLE_LAMBDA = 10.0
GP_LAMBDA = 10.0
TOTAL_SAMPLES = 1
AA_NUM_ATOMS = 1
CG_NUM_ATOMS = 1
NUM_DIMS = 3
NEIGHBORS = 1

#Dataset params
device=torch.device('cpu')
ngpu = 0
#device=torch.device('cuda:0')
#ngpu = 1
dataset_dir = 'data/'
cg_trajectory = 'CG.lammpstrj'
aa_trajectory = 'AA.lammpstrj'
cg_topology = 'CG.pdb'
aa_topology = 'AA.pdb'
output_size = 100
output_base = 'wgan_gp_outputs/'
output_name = sys.argv[1]
output_dir = output_base + output_name + '/'
output_traj_name = output_name + '.lammpstrj'
output_D_name = output_name + '_D.pth'
output_G_name = output_name + '_G.pth'
output_loss_name = output_name + "_losses.dat"
