import torch
import torch.nn as nn
import featurization as feat
#GENERATOR
class Generator(nn.Module):
    def __init__(self, ngpu, input_size, output_size, NUM_DIMS, G_WIDTH, G_DEPTH, Z_SIZE):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.NUM_DIMS = NUM_DIMS
        self.output_size = output_size
        self.input_size = input_size
        begin = [
                nn.Linear(Z_SIZE + (input_size * NUM_DIMS),G_WIDTH, bias=False),
                nn.BatchNorm1d(G_WIDTH),
                nn.LeakyReLU(0.2, inplace=True)
                ]
        linear = [
                nn.Linear(G_WIDTH,G_WIDTH, bias=False),
                nn.BatchNorm1d(G_WIDTH),
                nn.LeakyReLU(0.2, inplace=True)
                ]
        end = [
                nn.Linear(G_WIDTH, output_size* NUM_DIMS, bias=False),
                nn.Tanh()
                ]

        middle = []
        [middle.extend(linear) for i in range(G_DEPTH)]

        self.main = nn.Sequential(*begin + middle + end)

#Network takes two inputs, one is a vector of random numbers of shape (z)
#the second are the CG coordinates, shape (20,3)
    def forward(self, z, cg):
        input = torch.cat((z, cg.view(-1, self.NUM_DIMS * self.input_size)), dim=1)
        output = self.main(input)
        return(output.view(-1, self.output_size, self.NUM_DIMS).contiguous())


#DISCRIMINATOR


class Distance_Discriminator(nn.Module):
    def __init__(self, ngpu, input_size, D_WIDTH, D_DEPTH,  atom_counts, BATCH_SIZE, device):
        super(Distance_Discriminator, self).__init__()
        self.ngpu = ngpu
        self.atom_counts = atom_counts
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        
        begin = [
                nn.Linear(input_size, D_WIDTH, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                ]

        layer = [
                nn.Linear(D_WIDTH,D_WIDTH, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                ]

        end = [
                nn.Linear(D_WIDTH, 1, bias=False)
                ]

        middle = []
        [middle.extend(layer) for i in range(D_DEPTH)]

        self.main = nn.Sequential(*begin + middle + end)

    def forward(self, aa):
        features = feat.make_distance_set(aa, self.atom_counts, self.BATCH_SIZE, self.device)
        return self.main(features)


class Internal_Discriminator(nn.Module):
    def __init__(self, ngpu, input_size, D_WIDTH, D_DEPTH,  atom_counts, BATCH_SIZE, device):
        super(Internal_Discriminator, self).__init__()
        self.ngpu = ngpu
        self.atom_counts = atom_counts
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        
        begin = [
                nn.Linear(input_size, D_WIDTH, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                ]

        layer = [
                nn.Linear(D_WIDTH,D_WIDTH, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                ]

        end = [
                nn.Linear(D_WIDTH, 1, bias=False)
                ]

        middle = []
        [middle.extend(layer) for i in range(D_DEPTH)]

        self.main = nn.Sequential(*begin + middle + end)

    def forward(self, aa):
        return self.main(features)
