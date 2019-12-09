import torch
import torch.nn as nn
import featurization as feat
#GENERATOR
class Generator(nn.Module):
    def __init__(self, ngpu, CG_NUM_ATOMS, AA_NUM_ATOMS, NUM_DIMS, G_WIDTH, Z_SIZE):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.NUM_DIMS = NUM_DIMS
        self.AA_NUM_ATOMS = AA_NUM_ATOMS
        self.CG_NUM_ATOMS = CG_NUM_ATOMS
        self.main = nn.Sequential(
                nn.Linear(Z_SIZE + (CG_NUM_ATOMS * NUM_DIMS),G_WIDTH, bias=False),
                nn.BatchNorm1d(G_WIDTH),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(G_WIDTH,G_WIDTH, bias=False),
                nn.BatchNorm1d(G_WIDTH),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(G_WIDTH,G_WIDTH, bias=False),
                nn.BatchNorm1d(G_WIDTH),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(G_WIDTH,G_WIDTH, bias=False),
                nn.BatchNorm1d(G_WIDTH),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(G_WIDTH, AA_NUM_ATOMS * NUM_DIMS, bias=False),
                nn.Tanh(),
        )
#Network takes two inputs, one is a vector of random numbers of shape (z)
#the second are the CG coordinates, shape (20,3)
    def forward(self, z, cg):
        input = torch.cat((z, cg.view(-1, self.NUM_DIMS * self.CG_NUM_ATOMS)), dim=1)
        output = self.main(input)
        return(output.view(-1, self.AA_NUM_ATOMS, self.NUM_DIMS).contiguous())


#DISCRIMINATOR
class Discriminator(nn.Module):
    def __init__(self, ngpu, dist_size, D_WIDTH, atom_counts, BATCH_SIZE):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.atom_counts = atom_counts
        self.BATCH_SIZE = BATCH_SIZE
        self.main = nn.Sequential(
                nn.Linear(dist_size, D_WIDTH, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(D_WIDTH,D_WIDTH, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(D_WIDTH,D_WIDTH, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(D_WIDTH,D_WIDTH, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(D_WIDTH,D_WIDTH, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(D_WIDTH,D_WIDTH, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(D_WIDTH,D_WIDTH, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(D_WIDTH,D_WIDTH, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(D_WIDTH, 1, bias=False),
                #nn.Sigmoid(),
        )
    def forward(self, aa):
        features = feat.make_distance_set(aa, self.atom_counts, self.BATCH_SIZE)
        return self.main(features)


