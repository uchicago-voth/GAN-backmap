import torch
import torch.nn as nn
import featurization as feat
#GENERATOR
class Generator(nn.Module):
    def __init__(self, ngpu, CG_NUM_ATOMS, AA_NUM_ATOMS, NUM_DIMS, G_WIDTH, G_DEPTH, Z_SIZE):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.NUM_DIMS = NUM_DIMS
        self.AA_NUM_ATOMS = AA_NUM_ATOMS
        self.CG_NUM_ATOMS = CG_NUM_ATOMS
        begin = [
                nn.Linear(Z_SIZE + (CG_NUM_ATOMS * NUM_DIMS),G_WIDTH, bias=False),
                nn.BatchNorm1d(G_WIDTH),
                nn.LeakyReLU(0.2, inplace=True)
                ]
        linear = [
                nn.Linear(G_WIDTH,G_WIDTH, bias=False),
                nn.BatchNorm1d(G_WIDTH),
                nn.LeakyReLU(0.2, inplace=True)
                ]
        end = [
                nn.Linear(G_WIDTH, AA_NUM_ATOMS * NUM_DIMS, bias=False),
                nn.Tanh()
                ]

        middle = []
        [middle.extend(linear) for i in range(G_DEPTH)]

        self.main = nn.Sequential(*begin + middle + end)

#Network takes two inputs, one is a vector of random numbers of shape (z)
#the second are the CG coordinates, shape (20,3)
    def forward(self, z, cg):
        input = torch.cat((z, cg.view(-1, self.NUM_DIMS * self.CG_NUM_ATOMS)), dim=1)
        output = self.main(input)
        return(output.view(-1, self.AA_NUM_ATOMS, self.NUM_DIMS).contiguous())


#DISCRIMINATOR
class Discriminator(nn.Module):
    def __init__(self, ngpu, dist_size, D_WIDTH, D_DEPTH,  atom_counts, BATCH_SIZE):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.atom_counts = atom_counts
        self.BATCH_SIZE = BATCH_SIZE
        
        begin = [
                nn.Linear(dist_size, D_WIDTH, bias=False),
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
        features = feat.make_distance_set(aa, self.atom_counts, self.BATCH_SIZE)
        return self.main(features)


