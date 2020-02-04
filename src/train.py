from abc import ABC, abstractmethod
import loss_functions as loss
import torch
import numpy

class _Train(ABC):
    def __init__(self, forward_gen, forward_dis, dataset, aa_data_loader, cg_data_loader):
        self.f_gen = forward_gen
        self.f_dis = forward_dis
        self.data_loader = data_loader


    @abstractmethod
    def train(self, NUM_EPOCHS, D_TRAIN_RATE):
        for epoch in range(NUM_EPOCHS):
            print("epoch: " + str(epoch))
            for i, samples in enumerate(zip(cg_data_loader, aa_data_loader), 0):

        
        pass

    @abstractmethod
    def calc_cycle_loss():
        pass

    @abstractmethod
    def G_step():
        pass


    @abstractmethod
    def D_step():
        pass

class Distance_Trainer(_Trainer):
    def __init__(self, forward_gen, forward,dis, G_f, dataset):
        super().__init__()
        self.G_f = G_f





class Internal_Trainer(_Trainer):
    def __init__(self, forward_gen, backward_gen, forward_dis, backward_dis, dataset)
    self.b_gen = backward_gen



    def
