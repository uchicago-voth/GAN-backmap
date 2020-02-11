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


print("Constructing Dataset")
if c.mode == 0:
    dset = data.Cartesian_Dataset_Constructor(c)
elif c.mode == 1:
    dset = data.Internal_Dataset_Constructor(c)

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
        b_gen = torch.load("wgan_gp_outputs/" + input_model_name + '/' + input_model_name + '_G.pth')
        b_dis = torch.load("wgan_gp_outputs/" + input_model_name + '/' + input_model_name + '_D.pth')
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


elif mode == 1:
    if prev_model_flag:
        #load previous model
        print("loading model " + sys.argv[2])
        input_model_name = sys.argv[2]
        b_gen = torch.load("wgan_gp_outputs/" + input_model_name + '/' + input_model_name + '_G.pth')
        b_dis = torch.load("wgan_gp_outputs/" + input_model_name + '/' + input_model_name + '_D.pth')
        f_gen = torch.load("wgan_gp_outputs/" + input_model_name + '/' + 'forward_' + input_model_name + '_G.pth')
        f_dis = torch.load("wgan_gp_outputs/" + input_model_name + '/' + 'forward_' + input_model_name + '_D.pth')
    else:
        #Generate new models
        #GENERATORS
        b_gen = networks.Generator(c.CG_NUM_ATOMS, c.AA_NUM_ATOMS, c).to(c.device)
        b_gen.apply(init.weights_init_G)
        
        
        f_gen = networks.Generator(c.AA_NUM_ATOMS, c.CG_NUM_ATOMS, c).to(c.device)
        f_gen.apply(init.weights_init_G)
    
        #DISCRIMINATOR
        b_dis = networks.Internal_Discriminator(c.AA_NUM_ATOMS, c).to(c.device)
        b_dis.apply(init.weights_init_D)


        f_dis = networks.Internal_Discriminator(c.CG_NUM_ATOMS, c).to(c.device)
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
########################
#####train networks#####
########################

print("Beginning Training")

trainer.train(loss_file, c)









##labels
#real_label=1
#fake_label=-1
#norm_loss = nn.L1Loss()
#
#if c.NORM == 2:
#    norm_loss = nn.MSELoss()
#
#G_losses = []
#D_losses = []
#
#iters = 0
#
#print("Starting Training")
#for epoch in range(c.NUM_EPOCHS):
#    print("epoch: " + str(epoch))
#    for i, samples in enumerate(zip(cg_data_loader, aa_data_loader), 0):
#        
#        #Get Data
#        cg_real = samples[0][0].float().to(c.device).view(c.BATCH_SIZE, c.CG_NUM_ATOMS, c.NUM_DIMS)
#        aa_real = samples[1][0].float().to(c.device).view(c.BATCH_SIZE, c.AA_NUM_ATOMS, c.NUM_DIMS)
#        b_size = aa_real.size(0)
#        label = torch.full((b_size,), real_label, device=c.device)
#        
#
#        netD.zero_grad()
#        #Train with real
#        #forward thru Disc
#        output = netD(aa_real).view(-1)
#        #calc loss
#        errD_real =  loss.wasserstein_loss(output, label)
#        #calc gradients
#        errD_real.backward()
#        D_x = output.mean().item()
#        
#
#        #Train with all fake
#        noise = torch.randn(b_size, c.Z_SIZE, device=c.device)
#        aa_fake = netG(noise, cg_real)
#        label.fill_(fake_label)
#        #Forward
#        output = netD(aa_fake.detach()).view(-1)
#        #loss
#        errD_fake = loss.wasserstein_loss(output, label)
#        #gradients
#        errD_fake.backward()
#        D_G_z1 = output.mean().item()
#        
#
#        #Add real and fake gradients
#        gradient_penalty = loss.calc_gradient_penalty(netD, aa_real, aa_fake, cg_real, c)
#        gradient_penalty.backward()
#        errD = errD_real + errD_fake
#        
#        #update D
#        optimizerD.step()
#
#
#        #Train G less
#        if iters % c.D_TRAIN_RATE == 0:
#
#            #Update G
#            netG.zero_grad()
#            label.fill_(real_label)
#            #Run fake batch through D
#            output = netD(aa_fake).view(-1)
#            #loss
#            errG = loss.wasserstein_loss(output, label)
#            
#            err_cycle = loss.forward_cycle_loss(G_f, netG, aa_real, noise, c.CYCLE_LAMBDA) + loss.backward_cycle_loss(G_f, netG, cg_real, noise, c.CYCLE_LAMBDA)
#            errG_cycle = errG + err_cycle
#            #gradients
#            errG_cycle.backward()
#            D_G_z2 = output.mean().item()
#
#
#            #update G
#            optimizerG.step()
#            
#
#        iters += 1
#
#
#        #print stats
#        out_frequency = int(len(aa_data_loader)/5)
#        if (i % out_frequency == 0):
#
#            loss_file.write('[%d/%d][%d/%d]\tLoss_D: %.8f\tLoss_G: %.8f\tCycle_loss: %.8f\tD(x): %.8f\tD(G(z)): %.8f / %.8f\n'
#              % (epoch, c.NUM_EPOCHS, i, len(aa_data_loader), errD.item(), errG.item(), err_cycle, D_x, D_G_z1, D_G_z2))
#            loss_file.flush()
#
#        if (i == 0):
#            G_losses.append(errG.item())
#            D_losses.append(errD.item())
#    
#    if (epoch % c.model_output_freq == 0):
#        print('Saving model to ' + c.output_dir + ' at Epoch ' + str(epoch))
#        torch.save(netD, c.output_dir + c.output_D_name)
#        torch.save(netG, c.output_dir + c.output_G_name)



######################
#####save stuff#######
######################



print(c.output_size)
#generate LAMMPS trajectory file from generated samples for analysis
coords = dset.cg_trj[0:c.output_size]
normed_coords = dset.norm(coords)
cg_samples = torch.from_numpy(normed_coords)
verification_set = cg_samples[0:c.output_size,:,:].to(c.device)
big_noise = torch.randn(c.output_size, c.Z_SIZE, device=c.device)
#verification_data = torch.cat((big_noise, verification_set), dim=1)
samps = b_gen(big_noise, verification_set).to(c.device).detach()
samps = dset.unnorm(samps.cpu())

trj2 = md.load(c.dataset_dir + c.aa_trajectory, top = c.dataset_dir + c.aa_topology).atom_slice(range(c.AA_NUM_ATOMS), inplace=True)[0:c.output_size] 

#trj2 = dset.aa_trj.atom_slice(range(c.AA_NUM_ATOMS), inplace=True)[0:c.output_size]

#Save stuff
trj2.xyz = samps.numpy()
trj2.save_lammpstrj(c.output_dir + c.output_traj_name)

trainer.save_models('end', c)

#torch.save(netD, c.output_dir + c.output_D_name)
#torch.save(netG, c.output_dir + c.output_G_name)

#losses = open(output_dir + output_loss_name, "w")
#G_loss_array = np.asarray(G_losses)
#D_loss_array = np.asarray(D_losses)
#loss_array = np.swapaxes(np.vstack((G_loss_array, D_loss_array)), 0, 1)
#np.savetxt(c.output_dir+c.output_loss_name, loss_array, delimiter=' ')




