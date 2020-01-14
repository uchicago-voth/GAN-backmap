import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
sys.path.insert(0,'/home/loose/source/GAN-backmap/src/')

import config as c
import initialization as init
import featurization as feat
import networks
import inout as io
import loss_functions as loss
import data 




############################
#Read input files and parse#
############################

try:
    os.mkdir(c.output_dir)
except OSError:
    print("Directory already exists")
else:
    print("Created new directory")

print("Checking for input file")
if (os.path.exists(sys.argv[1] + ".in")):
    print("Input file found.")
    io.read_input_file(sys.argv[1] + ".in")

print("Device:")
print(c.device)


loss_file = open(c.output_dir + c.output_name + ".losses", "w")

########################
#####Create dataset#####
########################


print("Constructing Dataset")
dset = data.dataset_constructor(c.BATCH_SIZE, c.TOTAL_SAMPLES, c.dataset_dir, c.cg_trajectory, c.aa_trajectory, c.cg_topology, c.aa_topology)
c.AA_NUM_ATOMS = dset.AA_NUM_ATOMS
c.CG_NUM_ATOMS = dset.CG_NUM_ATOMS

aa_data_loader = dset.construct_data_loader(dset.aa_trj, c.AA_NUM_ATOMS)
cg_data_loader = dset.construct_data_loader(dset.cg_trj, c.CG_NUM_ATOMS)

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



#aa_data_loader, cg_data_loader, aa_trj, cg_samples, c.CG_NUM_ATOMS, c.AA_NUM_ATOMS, min_data, max_data = data.construct_dataset(
#        c.BATCH_SIZE, c.TOTAL_SAMPLES, c.dataset_dir, c.cg_trajectory, c.aa_trajectory, c.cg_topology, c.aa_topology)



print("Constructing Forward mapping matrix")
#G_f, atom_counts = data.forward_matrix(c.AA_NUM_ATOMS, c.CG_NUM_ATOMS, aa_trj, c.device)
dist_size = feat.calc_dist_size(atom_counts) 


##########################
####Initialize D and G####
##########################



prev_model_flag = False
if len(sys.argv) >= 3:
    prev_model_flag = True


if prev_model_flag:
    #load previous model
    print("loading model " + sys.argv[2])
    input_model_name = sys.argv[2]
    netG = torch.load("wgan_gp_outputs/" + input_model_name + '/' + input_model_name + '_G.pth')
    netD = torch.load("wgan_gp_outputs/" + input_model_name + '/' + input_model_name + '_D.pth')
else:
    #Generate new models
    #GENERATOR
    netG = networks.Generator(c.ngpu, c.CG_NUM_ATOMS, c.AA_NUM_ATOMS, c.NUM_DIMS, c.G_WIDTH, c.G_DEPTH, c.Z_SIZE).to(c.device)
    #if (device.type == 'cuda') and (ngpu > 1):
    #    netG = nn.DataParallel(netG, list(range(ngpu)))
    netG.apply(init.weights_init_G)

    #DISCRIMINATOR
    netD = networks.Discriminator(c.ngpu, dist_size, c.D_WIDTH, c.D_DEPTH, atom_counts, c.BATCH_SIZE, c.device).to(c.device)
    #if (device.type == 'cuda') and (ngpu > 1):
    #    netD = nn.DataParallel(netD, list(range(ngpu)))
    netD.apply(init.weights_init_D)


optimizerD = optim.RMSprop(netD.parameters(), lr=c.DLEARNING_RATE)

optimizerG = optim.RMSprop(netG.parameters(), lr=c.GLEARNING_RATE)

#Print models
print(netG)
print(netD)



########################
#####train networks#####
########################

#labels
real_label=1
fake_label=-1
norm_loss = nn.L1Loss()

if c.NORM == 2:
    norm_loss = nn.MSELoss()

G_losses = []
D_losses = []

iters = 0

print("Starting Training")
for epoch in range(c.NUM_EPOCHS):
    print("epoch: " + str(epoch))
    for i, samples in enumerate(zip(cg_data_loader, aa_data_loader), 0):
        
        #Get Data
        netD.zero_grad()
        cg_real = samples[0][0].float().to(c.device).view(c.BATCH_SIZE, c.CG_NUM_ATOMS, c.NUM_DIMS)
        aa_real = samples[1][0].float().to(c.device).view(c.BATCH_SIZE, c.AA_NUM_ATOMS, c.NUM_DIMS)
        b_size = aa_real.size(0)
        label = torch.full((b_size,), real_label, device=c.device)
        

        #Train with real
        #forward thru Disc
        output = netD(aa_real).view(-1)
        #calc loss
        errD_real =  loss.wasserstein_loss(output, label)
        #calc gradients
        errD_real.backward()
        D_x = output.mean().item()
        

        #Train with all fake
        noise = torch.randn(b_size, c.Z_SIZE, device=c.device)
        aa_fake = netG(noise, cg_real)
        label.fill_(fake_label)
        #Forward
        output = netD(aa_fake.detach()).view(-1)
        #loss
        errD_fake = loss.wasserstein_loss(output, label)
        #gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        

        #Add real and fake gradients
        gradient_penalty = loss.calc_gradient_penalty(netD, aa_real, aa_fake, cg_real, c.BATCH_SIZE, c.AA_NUM_ATOMS, c.CG_NUM_ATOMS, c.NUM_DIMS, c.GP_LAMBDA, c.device)
        gradient_penalty.backward()
        errD = errD_real + errD_fake
        
        #update D
        optimizerD.step()


        #Train G less
        if iters % c.D_TRAIN_RATE == 0:

            #Update G
            netG.zero_grad()
            label.fill_(real_label)
            #Run fake batch through D
            output = netD(aa_fake).view(-1)
            #loss
            errG = loss.wasserstein_loss(output, label)
            
            err_cycle = loss.forward_cycle_loss(norm_loss, G_f, netG, aa_real, noise, c.CYCLE_LAMBDA) + loss.backward_cycle_loss(norm_loss, G_f, netG, cg_real, noise, c.CYCLE_LAMBDA)
            errG_cycle = errG + err_cycle
            #gradients
            errG_cycle.backward()
            D_G_z2 = output.mean().item()


            #update G
            optimizerG.step()
            

        iters += 1


        #print stats
        out_frequency = int(len(aa_data_loader)/5)
        if (i % out_frequency == 0):

            loss_file.write('[%d/%d][%d/%d]\tLoss_D: %.8f\tLoss_G: %.8f\tCycle_loss: %.8f\tD(x): %.8f\tD(G(z)): %.8f / %.8f\n'
              % (epoch, c.NUM_EPOCHS, i, len(aa_data_loader), errD.item(), errG.item(), err_cycle, D_x, D_G_z1, D_G_z2))
            loss_file.flush()

        if (i == 0):
            G_losses.append(errG.item())
            D_losses.append(errD.item())
    
    if (epoch % c.model_output_freq == 0):
        print('Saving model to ' + c.output_dir + ' at Epoch ' + str(epoch))
        torch.save(netD, c.output_dir + c.output_D_name)
        torch.save(netG, c.output_dir + c.output_G_name)



######################
#####save stuff#######
######################




#generate LAMMPS trajectory file from generated samples for analysis
coords = dset.cg_trj.xyz[0:c.output_size]
normed_coords = dset.norm(coords)
cg_samples = torch.from_numpy(normed_coords)
verification_set = cg_samples[0:c.output_size,:,:].to(c.device)
big_noise = torch.randn(c.output_size, c.Z_SIZE, device=c.device)
#verification_data = torch.cat((big_noise, verification_set), dim=1)
samps = netG(big_noise, verification_set).to(c.device).detach()
samps = dset.unnorm(samps.cpu())

trj2 = dset.aa_trj.atom_slice(range(c.AA_NUM_ATOMS), inplace=True)[0:c.output_size]

#Save stuff
trj2.xyz = samps.numpy()
trj2.save_lammpstrj(c.output_dir + c.output_traj_name)
torch.save(netD, c.output_dir + c.output_D_name)
torch.save(netG, c.output_dir + c.output_G_name)

#losses = open(output_dir + output_loss_name, "w")
G_loss_array = np.asarray(G_losses)
D_loss_array = np.asarray(D_losses)
loss_array = np.swapaxes(np.vstack((G_loss_array, D_loss_array)), 0, 1)
np.savetxt(c.output_dir+c.output_loss_name, loss_array, delimiter=' ')




