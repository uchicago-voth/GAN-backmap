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
- A mapping correspondance file (Not necessary if you wish to backmap a protein at the residue level)
### Trajectories
- The trajectories should correspond to each other, but in theory this is not necessary. A cyclical loss is imposed on the generator
to penalize configurations which do not re-map to their original mapped state, but paired data will speed up training

- Provide each trajectory, as well as topology files for each, in a separate directory named "data/" (this name can be changed in the input file if desired)
- More sampling is always preferable to less, as the networks benefit from seing many examples of appropriate fine grained configurations

## 3. Examples
## 4. Dependencies
- Pytorch Version 1.0 or greater
- Numpy
- MDTraj 1.9.3 or greater
- Pandas


## 5. Current Limitations
- Currently only center of mass mappings are accepted
- Network depth is constant
- 


