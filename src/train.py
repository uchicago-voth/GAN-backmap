from abc import ABC, abstractmethod
import loss_functions as loss
import torch
import numpy
import torch.nn as nn





class _Train(ABC):
    def __init__(self, forward_gen, forward_dis, aa_data_loader, cg_data_loader, device, BATCH_SIZE, NUM_DIMS, CG_NUM_ATOMS, AA_NUM_ATOMS):
        self.f_gen = forward_gen
        self.f_dis = forward_dis
        self.aa_data_loader = aa_data_loader
        self.cg_data_loader = cg_data_loader
        self.device = device
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_DIMS = NUM_DIMS
        self.CG_NUM_ATOMS = CG_NUM_ATOMS
        self.AA_NUM_ATOMS = AA_NUM_ATOMS
        self.model_output_freq = model_output_freq


    
    def train(self, NUM_EPOCHS, D_TRAIN_RATE, NORM):
        real_label = 1
        fake_label = -1
        norm_loss = nn.L1Loss()
        if norm == 2:
            norm_loss = nn.MSELoss()

        iters = 0
        for epoch in range(NUM_EPOCHS):
            print("epoch: " + str(epoch))
            for i, samples in enumerate(zip(self.cg_data_loader, self.aa_data_loader), 0):
                cg_real = samples[0][0].float.to(self.device).view(self.BATCH_SIZE, self.CG_NUM_ATOMS, self.NUM_DIMS)
                aa_real = samples[0][0].float.to(self.device).view(self.BATCH_SIZE, self.AA_NUM_ATOMS, self.NUM_DIMS)
                b_size = aa_real.size()
                label = torch.full((b_size,), real_label, device=self.device)
                
                
                D_step()
                if iters % D_TRAIN_RATE == 0:
                    G_step()
        iters += 1
        out_frequency = int(len(self.aa_data_loader)/5)
        if i % out_frequency == 0:
            print_stats()
        if epoch % self.model_output_freq == 0:
            save_models()
        

    @abstractmethod
    def calc_cycle_loss():
        pass

    @abstractmethod
    def G_step():
        pass


    @abstractmethod
    def D_step():
        pass

    @abstractmethod
    def print_stats():
        pass

class Distance_Trainer(_Trainer):
    def __init__(self, forward_gen, forward_dis, G_f,  aa_data_loader, cg_data_loader, device, BATCH_SIZE, NUM_DIMS, CG_NUM_ATOMS, AA_NUM_ATOMS):
        super().__init__(self, forward_gen, forward_dis, aa_data_loader, cg_data_loader, device, BATCH_SIZE, NUM_DIMS, CG_NUM_ATOMS, AA_NUM_ATOMS):
        self.G_f = G_f





class Internal_Trainer(_Trainer):
    def __init__(self, forward_gen, backward_gen, forward_dis, backward_dis, dataset):
    def __init__(self, forward_gen, forward_dis, backward_gen, backward_dis, aa_data_loader, cg_data_loader, device, BATCH_SIZE, NUM_DIMS, CG_NUM_ATOMS, AA_NUM_ATOMS):
        super().__init__(self, forward_gen, forward_dis, aa_data_loader, cg_data_loader, device, BATCH_SIZE, NUM_DIMS, CG_NUM_ATOMS, AA_NUM_ATOMS):
        self.b_gen = backward_gen
        self.b_dis = backward_dis
    
    def D_step()


    def calc_cycle_loss(self, aa, cg)
