import torch
import torch.nn as nn
import featurization as feat
#GENERATOR
class Generator(nn.Module):
    def __init__(self, input_size, output_size, param):
        super(Generator, self).__init__()
        self.device = param.device
        self.ngpu = param.ngpu
        self.NUM_DIMS = param.NUM_DIMS
        self.output_size = output_size
        self.input_size = input_size
        self.WIDTH = param.G_WIDTH
        self.DEPTH = param.G_DEPTH
        self.Z_SIZE = param.Z_SIZE
        begin = [
                nn.Linear(self.Z_SIZE + (input_size * self.NUM_DIMS),self.WIDTH, bias=False),
                nn.BatchNorm1d(self.WIDTH),
                nn.LeakyReLU(0.2, inplace=True)
                ]
        linear = [
                nn.Linear(self.WIDTH,self.WIDTH, bias=False),
                nn.BatchNorm1d(self.WIDTH),
                nn.LeakyReLU(0.2, inplace=True)
                ]
        end = [
                nn.Linear(self.WIDTH, output_size* self.NUM_DIMS, bias=False),
                nn.Tanh()
                ]

        middle = []
        [middle.extend(linear) for i in range(self.DEPTH)]

        self.main = nn.Sequential(*begin + middle + end)

#Network takes two inputs, one is a vector of random numbers of shape (z)
#the second are the CG coordinates, shape (20,3)
    def forward(self, z, cg):
        input = torch.cat((z, cg.view(-1, self.NUM_DIMS * self.input_size)), dim=1)
        output = self.main(input)
        return(output.view(-1, self.output_size, self.NUM_DIMS).contiguous())


#DISCRIMINATOR


class Distance_Discriminator(nn.Module):
    def __init__(self, input_size, atom_counts, param):
        super(Distance_Discriminator, self).__init__()
        self.ngpu = param.ngpu
        self.atom_counts = atom_counts
        self.BATCH_SIZE = param.BATCH_SIZE
        self.device = param.device
        self.input_size = input_size
        self.WIDTH = param.D_WIDTH
        self.DEPTH = param.D_DEPTH

        begin = [
                nn.Linear(self.input_size, self.WIDTH, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                ]

        layer = [
                nn.Linear(self.WIDTH,self.WIDTH, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                ]

        end = [
                nn.Linear(self.WIDTH, 1, bias=False)
                ]

        middle = []
        [middle.extend(layer) for i in range(self.DEPTH)]

        self.main = nn.Sequential(*begin + middle + end)

    def forward(self, aa):
        features = feat.make_distance_set(aa, self.atom_counts, self.BATCH_SIZE, self.device)
        return self.main(features)


class Internal_Discriminator(nn.Module):
    def __init__(self, input_size, param):
        super(Internal_Discriminator, self).__init__()
        self.ngpu = param.ngpu 
        self.BATCH_SIZE = param.BATCH_SIZE
        self.device = param.device
        self.input_size = input_size
        self.WIDTH = param.D_WIDTH
        self.DEPTH = param.D_DEPTH
        
        begin = [
                nn.Linear(input_size, self.WIDTH, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                ]

        layer = [
                nn.Linear(self.WIDTH,self.WIDTH, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                ]

        end = [
                nn.Linear(self.WIDTH, 1, bias=False)
                ]

        middle = []
        [middle.extend(layer) for i in range(self.DEPTH)]

        self.main = nn.Sequential(*begin + middle + end)

    def forward(self, data):
        data = data.view(self.BATCH_SIZE, self.input_size)
        return self.main(data)
