from abc import ABC, abstractmethod
import loss_functions as loss
import torch
import numpy
import torch.nn as nn
import networks



class _Train(ABC):
    def __init__(self, forward_gen, forward_dis, aa_data_loader, cg_data_loader, param):
        

        self.f_gen = forward_gen
        self.f_dis = forward_dis
        self.aa_data_loader = aa_data_loader
        self.cg_data_loader = cg_data_loader
        self.real_label = 1
        self.fake_label = -1


    
    def train(self, loss_file, param):
        iters = 0
        for epoch in range(param.NUM_EPOCHS):
            print("epoch: " + str(epoch))
            for i, samples in enumerate(zip(self.cg_data_loader, self.aa_data_loader), 0):
                cg_real = samples[0][0].float.to(param.device).view(param.BATCH_SIZE, param.CG_NUM_ATOMS, param.NUM_DIMS)
                aa_real = samples[0][0].float.to(self.device).view(param.BATCH_SIZE, param.AA_NUM_ATOMS, param.NUM_DIMS)
                b_size = aa_real.size()
                label = torch.full((b_size,), real_label, device=param.device)
                
                G_TRAIN = iters % param.D_TRAIN_RATE
                errors = self.step(aa_real, cg_real, label, b_size, param) GP_LAMBDA, CYCLE_LAMBDA, G_TRAIN)
                iters += 1
                out_frequency = int(len(self.aa_data_loader)/5)
                if i % out_frequency == 0:
                    self.print_stats(errors, epoch, param.NUM_EPOCHS, loss_file)
                if epoch % self.model_output_freq == 0:
                    self.save_models()
        

    @abstractmethod
    def calc_cycle_loss():
        pass

    @abstractmethod
    def step():
        pass

    @abstractmethod
    def print_stats():
        pass

    @abstractmethod
    def save_models():
        pass

class Distance_Trainer(_Trainer):
    def __init__(self, forward_gen, forward_dis, G_f,  aa_data_loader, cg_data_loader, param):
        super().__init__(self, forward_gen, forward_dis, aa_data_loader, cg_data_loader, param):
        self.G_f = G_f


    def step(self, aa_real, cg_real, label, b_size, param): GP_LAMBDA, CYCLE_LAMBDA, G_TRAIN):
        self.f_dis.zero_grad()
        output = self.f_dis(aa_real).view(-1)

        errD_real = loss.wasserstein_loss(output, label)
        errD_real.backward()
        noise = torch.randn(b_size, self.Z_SIZE, device=self.device)
        aa_fake = self.f_gen(noise, cg_real)
        label.fill_(fake_label)
        output = self.f_dis(aa_fake.detach()).view(-1)

        errD_fake = loss.wasserstein_loss(output_label)
        errD_fake.backward()


        gradient_penalty = loss.calc_gradient_penalty(f_dis, aa_real, aa_fake, cg_real, param)

        gradient_penalty.backward()
        errD = errD_fake + errD_real
        
        optimizerD.step()
        if G_TRAIN == 0:

            self.f_gen.zero_grad()
            label.fill_(real_label)
            output = self.f_dis(aa_fake)
            errG = loss.wasserstein_loss(output_label)
            
            
            err_cycle = self.calc_cycle_loss(self.G_f, self.f_gen, noise, param.CYCLE_LAMBDA)
            errG_cycle = errG + err_cycle
            errG_cycle.backward()
        
        return [errD, errG, err_cycle, gradient_penalty]
    

    def calc_cycle_loss(self, G_f, f_gen, aa_real, noise, CYCLE_LAMBDA):
        return loss.forward_cycle_loss(G_f, f_gen, aa_real, noise, CYCLE_LAMBDA) + loss.backward_cycle_loss(G_f, f_gen, cg_real, noise, CYCLE_LAMBDA)

    def print_stats(self, errD, errG, err_cycle, gradient_penalty, epoch, NUM_EPOCHS, loss_file):

        loss_file.write('[%d/%d][%d/%d]\tLoss_D: %.8f\tLoss_G: %.8f\tCycle_loss: %.8f\tGradient penalty: %.8f'
            % (epoch, NUM_EPOCHS, i, len(self.aa_data_loader), errD.item(), errG.item(), err_cycle, gradient_penalty))
        loss_file.flush()

class Internal_Trainer(_Trainer):
    def __init__(self, forward_gen, forward_dis, backward_gen, backward_dis, aa_data_loader, cg_data_loader, param):
        super().__init__(self, forward_gen, forward_dis, aa_data_loader, cg_data_loader, param):
        self.b_gen = backward_gen
        self.b_dis = backward_dis
    
    def D_step()


    def calc_cycle_loss(self, aa, cg)
