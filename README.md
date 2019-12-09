# GAN-backmap

GAN-backmap is a method to reconstruct fine-grained configurations of molecular data from coarse-grained ones.
The method uses generative Adversarial Networks to learn the distributions of atomic positions conditioned on the 
positions of the coarse-grained sites.

## 1. How the method works
The networks are trained on two sets of data, one being an atomistic/fine-grained trajectory, and the other being 
a mapped coarse-grained version of the fine grained trajectory. A generator network (which starts randomly initialized,
attempts to make fine grained configurations given a random vector and a coarse grained configurations. A critic
is shown both the original fine grain configurations and generated ones, and is trained to distinguish between the two.
The generator is trained to generate samples that the critic cannot distinguish.


## 2. How to use the code
Using the method requires # things:
- A fine grained trajectory and a mapped trajectory
- An input file
- A mapping correspondance file


## 3. Examples
## 4. Dependencies
Pytorch Version 1.0 or greater
Numpy
MDTraj
Pandas





