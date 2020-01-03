import torch
import numpy as np


#Used to calculate the total size of the discriminator input
def calc_dist_size(atom_counts):
    prev = 0
    count = 0
    for i in atom_counts:
        count += np.square((i-prev))
        prev=i
    return count

#Construct a gram matrix given coordinates
def coords_to_gram_matrix(points):
    G = torch.matmul(points, torch.transpose(points, 1, 2))
    return G

#Construct a Euclidean Distance Matrix from a Gram Matrix
def gram_to_EDM(G, BATCH_SIZE):
    D = torch.diagonal(G, dim1=1, dim2=2).contiguous().view(BATCH_SIZE ,1, -1) + torch.transpose(torch.diagonal(G, dim1=1, dim2=2).contiguous().view(BATCH_SIZE, 1,-1), 1, 2) - 2*G
    return D


#Takes total EDM and then uses N off-diagonals to construct a set of distances. This is slow and bad and bad and slow
def make_neigh_list(matrix, neighbors):
    l = torch.zeros((0), dtype=torch.float).to(device)
    for i in range(neighbors):
        l = torch.cat((l, torch.diagonal(matrix, offset=i+1, dim1=0, dim2=1)), dim=0)
    return l


#Featurization options, first is best

#Construct a set of distance matrices based on the mapping of the system. This is decently efficient
def make_distance_set(coords, atom_counts, BATCH_SIZE, device):
    prev = 0
    dists = torch.zeros((BATCH_SIZE, 0)).to(device)
    for i in atom_counts:
        coords_segment = coords[:,prev:i,:]
        prev=i
        D = gram_to_EDM(coords_to_gram_matrix(coords_segment), BATCH_SIZE).view(BATCH_SIZE, -1)
        dists = torch.cat((dists, D), dim=1)
    return dists


#Constructs a feature set using off-diagonals. As commented earlier, this is slow and cumbersome
def featurize(coords, neighbors):
    for i in range(list(coords.shape)[0]):
        G = coords_to_gram_matrix(coords[i,:,:])
        D = gram_to_EDM(G)
        l = make_neigh_list(D, neighbors)
        if i == 0:
            features = l.view(1, -1)
        else:
            features = torch.cat((features, l.view(1,-1)), dim=0)
    return features




