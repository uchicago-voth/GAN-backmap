import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import mdtraj as md

sys.path.insert(0,'/home/loose/source/GAN-backmap/src/')

#import config as c
import initialization as init
import featurization as feat
import networks
import parameters as param
import loss_functions as loss
import data 
import train




############################
#Read input files and parse#
############################

c = param.config()

try:
    os.mkdir(c.output_dir)
except OSError:
    print("Directory already exists")
else:
    print("Created new directory")

print("Checking for input file")
if (os.path.exists(sys.argv[1] + ".in")):
    print("Input file found.")
    c.read_input_file(sys.argv[1] + ".in")

print("Device:")
print(c.device)


loss_file = open(c.output_dir + c.output_name + ".losses", "w")

########################
#####Create dataset#####
########################

print(c.mode)
print("Constructing Dataset")
if c.mode == 0:
    dset = data.Cartesian_Dataset_Constructor(c)
elif c.mode == 1:
    dset = data.Internal_Dataset_Constructor(c)
elif c.mode == 2:
    dset = data.Internal_Fragment_Dataset_Constructor(c)


c.AA_NUM_ATOMS = dset.AA_NUM_ATOMS
c.CG_NUM_ATOMS = dset.CG_NUM_ATOMS

aa_data_loader = dset.construct_data_loader(dset.aa_trj, dset.AA_NUM_ATOMS)
cg_data_loader = dset.construct_data_loader(dset.cg_trj, dset.CG_NUM_ATOMS)


if c.mode == 0:
    atom_counts = []
    if (os.path.exists(sys.argv[1] + ".cg")):
        print("Atomistic-CG correspondance file located. Generating neighbor list and forward mapping operator based on file.")
        atom_counts = dset.input_atom_counts(sys.argv[1] + ".cg")
    else:
        print("No Atomistic-CG correspondance file located. Generating neighbor list and forward mapping operator based on single residue per bead mapping.")
        print("This option only works properly for proteins. If your system is not a protein, this may break things.")
        atom_counts = dset.res_atom_counts()
    
    
    
    print("Constructing Forward Matrix")
    G_f = dset.construct_forward_matrix(atom_counts, c.device)
    dist_size = feat.calc_dist_size(atom_counts) 


##########################
####Initialize D and G####
##########################



prev_model_flag = False
if len(sys.argv) >= 3:
    prev_model_flag = True

if c.mode == 0:
    if prev_model_flag:
        #load previous model
        print("loading model " + sys.argv[2])
        input_model_name = sys.argv[2]
        b_gen = torch.load(c.output_base + input_model_name + '/' + input_model_name + '_G.pth')
        b_dis = torch.load(c.output_base + input_model_name + '/' + input_model_name + '_D.pth')
    else:
        #Generate new models
        #GENERATOR
        b_gen = networks.Generator(c.CG_NUM_ATOMS, c.AA_NUM_ATOMS, c).to(c.device)
        b_gen.apply(init.weights_init_G)
    
        #DISCRIMINATOR
        b_dis = networks.Distance_Discriminator(dist_size, atom_counts, c).to(c.device)
        b_dis.apply(init.weights_init_D)
    
    
    optimizerD = optim.RMSprop(b_dis.parameters(), lr=c.DLEARNING_RATE)
    
    optimizerG = optim.RMSprop(b_gen.parameters(), lr=c.GLEARNING_RATE)
    
    #Print models
    print(b_gen)
    print(b_dis)

    trainer = train.Distance_Trainer(b_gen, b_dis, optimizerD, optimizerG, G_f, aa_data_loader, cg_data_loader, c)    


elif c.mode == 1:
    if prev_model_flag:
        #load previous model
        print("loading model " + sys.argv[2])
        input_model_name = sys.argv[2]
        b_gen = torch.load(c.output_base + input_model_name + '/' + input_model_name + '_G.pth')
        b_dis = torch.load(c.output_base + input_model_name + '/' + input_model_name + '_D.pth')
        f_gen = torch.load(c.output_base + input_model_name + '/' + 'forward_' + input_model_name + '_G.pth')
        f_dis = torch.load(c.output_base + input_model_name + '/' + 'forward_' + input_model_name + '_D.pth')
    else:
        #Generate new models
        #GENERATORS
        b_gen = networks.Generator(c.CG_NUM_ATOMS, c.AA_NUM_ATOMS, c).to(c.device)
        b_gen.apply(init.weights_init_G)
        
        
        f_gen = networks.Generator(c.AA_NUM_ATOMS, c.CG_NUM_ATOMS, c).to(c.device)
        f_gen.apply(init.weights_init_G)
    
        #DISCRIMINATOR
        b_dis = networks.Internal_Discriminator(c.AA_NUM_ATOMS * c.NUM_DIMS, c).to(c.device)
        b_dis.apply(init.weights_init_D)


        f_dis = networks.Internal_Discriminator(c.CG_NUM_ATOMS * c.NUM_DIMS, c).to(c.device)
        f_dis.apply(init.weights_init_D)
    
    
    optimizer_b_gen = optim.RMSprop(b_gen.parameters(), lr=c.GLEARNING_RATE)
    optimizer_f_gen = optim.RMSprop(f_gen.parameters(), lr=c.GLEARNING_RATE)
   
    optimizer_b_dis = optim.RMSprop(b_dis.parameters(), lr=c.DLEARNING_RATE)
    optimizer_f_dis = optim.RMSprop(f_dis.parameters(), lr=c.DLEARNING_RATE)
    
    
    #Print models
    print(b_dis)
    print(b_gen)

    print(f_dis)
    print(f_gen)

    trainer = train.Internal_Trainer(b_gen, b_dis, optimizer_b_gen, optimizer_b_dis, f_gen, f_dis, optimizer_f_gen, optimizer_f_dis, aa_data_loader, cg_data_loader, c)    

elif c.mode == 2:
    if prev_model_flag:
        #load previous model
        print("loading model " + sys.argv[2])
        input_model_name = sys.argv[2]
        b_gen = torch.load(c.output_base + input_model_name + '/' + input_model_name + '_G.pth')
        b_dis = torch.load(c.output_base + input_model_name + '/' + input_model_name + '_D.pth')
    else:
        #Generate new models
        #GENERATOR
        b_gen = networks.Generator(c.CG_NUM_ATOMS, c.AA_NUM_ATOMS, c).to(c.device)
        b_gen.apply(init.weights_init_G)
    
        #DISCRIMINATOR
        b_dis = networks.Internal_Discriminator(c.AA_NUM_ATOMS * c.NUM_DIMS, c).to(c.device)
        b_dis.apply(init.weights_init_D)
    
    
    optimizerD = optim.RMSprop(b_dis.parameters(), lr=c.DLEARNING_RATE)
    
    optimizerG = optim.RMSprop(b_gen.parameters(), lr=c.GLEARNING_RATE)
    
    #Print models
    print(b_gen)
    print(b_dis)

    trainer = train.Internal_Fragment_Trainer(b_gen, b_dis, optimizerD, optimizerG, aa_data_loader, cg_data_loader, c)    

########################
#####train networks#####
########################

print("Beginning Training")

trainer.train(loss_file, c)











######################
#####save stuff#######
######################



#generate LAMMPS trajectory file from generated samples for analysis
coords = dset.cg_trj[0:c.output_size]
normed_coords = dset.norm(coords)
cg_samples = torch.from_numpy(normed_coords).float()
verification_set = cg_samples[0:c.output_size,:,:].to(c.device)
big_noise = torch.randn(c.output_size, c.Z_SIZE, device=c.device)
#verification_data = torch.cat((big_noise, verification_set), dim=1)
samps = b_gen(big_noise, verification_set).to(c.device).detach()
samps = dset.unnorm(samps.cpu())


if c.mode == 0:
    trj2 = md.load(c.dataset_dir + c.aa_trajectory, top = c.dataset_dir + c.aa_topology).atom_slice(range(c.AA_NUM_ATOMS), inplace=True)[0:c.output_size] 
    
    #trj2 = dset.aa_trj.atom_slice(range(c.AA_NUM_ATOMS), inplace=True)[0:c.output_size]
    
    #Save stuff
    trj2.xyz = samps.numpy()
    trj2.save_lammpstrj(c.output_dir + c.output_traj_name)
elif c.mode == 1 or c.mode == 2:
    samps = samps.numpy().reshape(-1, 3)
    np.savetxt(c.output_dir + c.output_name + '.bad', samps, fmt= '%.6f', delimiter=',')
    if c.mode == 2:
        trj2 = md.load(c.dataset_dir + c.cg_trajectory, top=c.dataset_dir + c.cg_topology).atom_slice(range(c.CG_NUM_ATOMS), inplace=True)[0:c.output_size]
        trj2.xyz = coords
        trj2.save_lammpstrj(c.output_dir + c.output_name + "_ca.lammpstrj")
trainer.save_models('end', c)

#torch.save(netD, c.output_dir + c.output_D_name)
#torch.save(netG, c.output_dir + c.output_G_name)

#losses = open(output_dir + output_loss_name, "w")
#G_loss_array = np.asarray(G_losses)
#D_loss_array = np.asarray(D_losses)
#loss_array = np.swapaxes(np.vstack((G_loss_array, D_loss_array)), 0, 1)
#np.savetxt(c.output_dir+c.output_loss_name, loss_array, delimiter=' ')




