from abc import ABC, abstractmethod
import loss_functions as loss
import torch
import numpy
import torch.nn as nn
import networks



class _Trainer(ABC):
    def __init__(self, backward_gen, backward_dis, optimizer_b_gen, optimizer_b_dis, aa_data_loader, cg_data_loader, param):
        

        self.b_gen = backward_gen
        self.b_dis = backward_dis
        self.optimizer_b_gen = optimizer_b_gen
        self.optimizer_b_dis = optimizer_b_dis
        self.aa_data_loader = aa_data_loader
        self.cg_data_loader = cg_data_loader
        self.real_label = 1
        self.fake_label = -1
        self.Z_SIZE = param.Z_SIZE


    
    def train(self, loss_file, param):
        errors = self.errors
        iters = 0
        for epoch in range(param.NUM_EPOCHS):
            print("epoch: " + str(epoch))
            for i, samples in enumerate(zip(self.cg_data_loader, self.aa_data_loader), 0):
                cg_real = samples[0][0].float().to(param.device).view(param.BATCH_SIZE, param.CG_NUM_ATOMS, param.NUM_DIMS)
                aa_real = samples[1][0].float().to(param.device).view(param.BATCH_SIZE, param.AA_NUM_ATOMS, param.NUM_DIMS)
                b_size = aa_real.size(0)
                label = torch.full((b_size,), self.real_label, device=param.device)
                
                G_TRAIN = iters % param.D_TRAIN_RATE
                errors = self.step(aa_real, cg_real, label, b_size, G_TRAIN, errors, param)
                iters += 1
                out_frequency = int(len(self.aa_data_loader)/5)
                if i % out_frequency == 0:
                    self.print_stats(errors, epoch, param.NUM_EPOCHS, i, loss_file)
            if epoch % param.model_output_freq == 0:
                self.save_models(epoch, param)
        


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
    def __init__(self, backward_gen, backward_dis, optimizer_b_gen, optimizer_b_dis, G_f, aa_data_loader, cg_data_loader, param):
        super().__init__(backward_gen, backward_dis, optimizer_b_gen, optimizer_b_dis, aa_data_loader, cg_data_loader, param)
        self.G_f = G_f
        self.errors = [0,0,0,0]


    def step(self, aa_real, cg_real, label, b_size, G_TRAIN, errors, param):
        errD = errors[0]
        errG = errors[1]
        err_cycle = errors[2]
        gradient_penalty = errors[3]

        self.b_dis.zero_grad()
        output = self.b_dis(aa_real).view(-1)

        errD_real = loss.wasserstein_loss(output, label)
        errD_real.backward()
        noise = torch.randn(b_size, self.Z_SIZE, device=param.device)
        aa_fake = self.b_gen(noise, cg_real)
        label.fill_(self.fake_label)
        output = self.b_dis(aa_fake.detach()).view(-1)

        errD_fake = loss.wasserstein_loss(output, label)
        errD_fake.backward()


        gradient_penalty = loss.calc_gradient_penalty(self.b_dis, aa_real, aa_fake, cg_real, param.AA_NUM_ATOMS, param)

        gradient_penalty.backward()
        errD = errD_fake + errD_real
        
        self.optimizer_b_dis.step()
        if G_TRAIN == 0:

            self.b_gen.zero_grad()
            label.fill_(self.real_label)
            output = self.b_dis(aa_fake)
            errG = loss.wasserstein_loss(output, label)
            
            
            err_cycle = self.calc_cycle_loss(self.G_f, self.b_gen, aa_real, cg_real, noise, param.CYCLE_LAMBDA)
            errG_cycle = errG + err_cycle
            errG_cycle.backward()
            self.optimizer_b_gen.step()
        
        return [errD, errG, err_cycle, gradient_penalty]
    

    def calc_cycle_loss(self, G_f, b_gen, aa_real, cg_real, noise, CYCLE_LAMBDA):
        return loss.forward_cycle_loss(G_f, b_gen, aa_real, noise, CYCLE_LAMBDA) + loss.backward_cycle_loss(G_f, b_gen, cg_real, noise, CYCLE_LAMBDA)

    def print_stats(self, errors, epoch, NUM_EPOCHS, step, loss_file):
        errD = errors[0]
        errG = errors[1]
        err_cycle = errors[2]
        gradient_penalty = errors[3]
        loss_file.write('[%d/%d][%d/%d]\tLoss_D: %.8f\tLoss_G: %.8f\tCycle_loss: %.8f\tGradient penalty: %.8f\n'
            % (epoch, NUM_EPOCHS, step, len(self.aa_data_loader), errD.item(), errG.item(), err_cycle, gradient_penalty))
        loss_file.flush()

    def save_models(self, epoch, param):
        print('saving model to ' + param.output_dir + ' at Epoch ' + str(epoch))
        torch.save(self.b_dis, param.output_dir + param.output_D_name)
        torch.save(self.b_gen, param.output_dir + param.output_G_name)

class Internal_Trainer(_Trainer):
    def __init__(self, backward_gen, backward_dis, optimizer_b_gen, optimizer_b_dis, forward_gen, forward_dis, optimizer_f_gen, optimizer_f_dis, aa_data_loader, cg_data_loader, param):
        super().__init__(backward_gen, backward_dis, optimizer_b_gen, optimizer_b_dis, aa_data_loader, cg_data_loader, param)
        self.f_gen = forward_gen
        self.f_dis = forward_dis
        self.optimizer_f_gen = optimizer_f_gen
        self.optimizer_f_dis = optimizer_f_dis
        self.errors = [0,0,0,0,0,0,0]
    
    def step(self, aa_real, cg_real, label, b_size, G_TRAIN, errors, param):
        err_b_dis = errors[0]
        err_b_gen = errors[1]
        err_f_dis = errors[2]
        err_f_gen = errors[3]
        err_cycle = errors[4]
        b_gradient_penalty = errors[5]
        f_gradient_penalty = errors[6]


        self.b_dis.zero_grad()
        self.f_dis.zero_grad()

        aa_output = self.b_dis(aa_real).view(-1)
        cg_output = self.f_dis(cg_real).view(-1)

        err_b_dis_real = loss.wasserstein_loss(aa_output, label)
        err_f_dis_real = loss.wasserstein_loss(cg_output, label)

        err_b_dis_real.backward()
        err_f_dis_real.backward()

        noise = torch.randn(b_size, self.Z_SIZE, device=param.device)

        aa_fake = self.b_gen(noise, cg_real)
        cg_fake = self.f_gen(noise, aa_real)

        label.fill_(self.fake_label)

        aa_output = self.b_dis(aa_fake.detach()).view(-1)
        cg_output = self.f_dis(cg_fake.detach()).view(-1)

        err_b_dis_fake = loss.wasserstein_loss(aa_output, label)
        err_f_dis_fake = loss.wasserstein_loss(cg_output, label)

        err_b_dis_fake.backward()
        err_f_dis_fake.backward()

        b_gradient_penalty = loss.calc_gradient_penalty(self.b_dis, aa_real, aa_fake, cg_real, param.AA_NUM_ATOMS, param)
        f_gradient_penalty = loss.calc_gradient_penalty(self.f_dis, cg_real, cg_fake, aa_real, param.CG_NUM_ATOMS, param)

        b_gradient_penalty.backward()
        f_gradient_penalty.backward()

        err_b_dis = err_b_dis_fake + err_b_dis_real
        err_f_dis = err_f_dis_fake + err_f_dis_real

        self.optimizer_b_dis.step()
        self.optimizer_f_dis.step()


        if G_TRAIN == 0:

            self.b_gen.zero_grad()
            self.f_gen.zero_grad()
        
            label.fill_(self.real_label)

            aa_output = self.b_dis(aa_fake)
            cg_output = self.f_dis(cg_fake)
  
            err_b_gen = loss.wasserstein_loss(aa_output, label)
            err_f_gen = loss.wasserstein_loss(cg_output, label)
            
            err_b_gen.backward()
            err_f_gen.backward()

            err_cycle = self.calc_cycle_loss(self.f_gen, self.b_gen, aa_real, cg_real, noise, param.CYCLE_LAMBDA)
            err_cycle.backward()
            
            self.optimizer_b_gen.step()
            self.optimizer_f_gen.step()

        return [err_b_dis, err_b_gen, err_f_dis, err_b_dis, err_cycle, b_gradient_penalty, f_gradient_penalty]

    def calc_cycle_loss(self, f_gen, b_gen, aa_real, cg_real, noise, CYCLE_LAMBDA):
        return loss.general_cycle_loss(f_gen, b_gen, aa_real, noise, CYCLE_LAMBDA) + loss.general_cycle_loss(b_gen, f_gen, cg_real, noise, CYCLE_LAMBDA)



    def print_stats(self, errors, epoch, NUM_EPOCHS, step, loss_file):
        err_b_dis = errors[0]
        err_b_gen = errors[1]
        err_f_dis = errors[2]
        err_f_gen = errors[3]
        err_cycle = errors[4]
        b_gradient_penalty = errors[5]
        f_gradient_penalty = errors[6]
        loss_file.write('[%d/%d][%d/%d]\tBackward D loss: %.5f\tBackward G Loss: %.5f\tForward D loss: %.5f\t Foward G loss: %.5f\tCycle_loss: %.5f\tback gradient penalty: %.5f\tforward gradient penalty: %.5f\n'
                % (epoch, NUM_EPOCHS, step, len(self.aa_data_loader), err_b_dis.item(), err_b_gen.item(), 
                err_f_dis.item(), err_f_gen.item(), err_cycle, b_gradient_penalty, f_gradient_penalty))
        loss_file.flush()


    def save_models(self, epoch, param):
        print('saving model to ' + param.output_dir + ' at Epoch ' + str(epoch))
        torch.save(self.b_dis, param.output_dir + param.output_D_name)
        torch.save(self.b_gen, param.output_dir + param.output_G_name)
        torch.save(self.f_dis, param.output_dir + 'forward_' +  param.output_D_name)
        torch.save(self.f_gen, param.output_dir + 'forward_' +  param.output_G_name)
 




class Internal_Fragment_Trainer(_Trainer):
    def __init__(self, backward_gen, backward_dis, optimizer_b_gen, optimizer_b_dis, aa_data_loader, cg_data_loader, param):
        super().__init__(backward_gen, backward_dis, optimizer_b_gen, optimizer_b_dis, aa_data_loader, cg_data_loader, param)
        self.errors = [0,0,0]


    def step(self, aa_real, cg_real, label, b_size, G_TRAIN, errors, param):
        errD = errors[0]
        errG = errors[1]
        gradient_penalty = errors[2]

        self.b_dis.zero_grad()
        output = self.b_dis(aa_real).view(-1)

        errD_real = loss.wasserstein_loss(output, label)
        errD_real.backward()
        noise = torch.randn(b_size, self.Z_SIZE, device=param.device)
        aa_fake = self.b_gen(noise, cg_real)
        label.fill_(self.fake_label)
        output = self.b_dis(aa_fake.detach()).view(-1)

        errD_fake = loss.wasserstein_loss(output, label)
        errD_fake.backward()


        gradient_penalty = loss.calc_gradient_penalty(self.b_dis, aa_real, aa_fake, cg_real, param.AA_NUM_ATOMS, param)

        gradient_penalty.backward()
        errD = errD_fake + errD_real
        
        self.optimizer_b_dis.step()
        if G_TRAIN == 0:

            self.b_gen.zero_grad()
            label.fill_(self.real_label)
            output = self.b_dis(aa_fake)
            errG = loss.wasserstein_loss(output, label)
            errG.backward()

            self.optimizer_b_gen.step()
        
        return [errD, errG, gradient_penalty]
    


    def print_stats(self, errors, epoch, NUM_EPOCHS, step, loss_file):
        errD = errors[0]
        errG = errors[1]
        gradient_penalty = errors[2]
        loss_file.write('[%d/%d][%d/%d]\tLoss_D: %.8f\tLoss_G: %.8f\tGradient penalty: %.8f\n'
            % (epoch, NUM_EPOCHS, step, len(self.aa_data_loader), errD.item(), errG.item(), gradient_penalty))
        loss_file.flush()

    def save_models(self, epoch, param):
        print('saving model to ' + param.output_dir + ' at Epoch ' + str(epoch))
        torch.save(self.b_dis, param.output_dir + param.output_D_name)
        torch.save(self.b_gen, param.output_dir + param.output_G_name)




