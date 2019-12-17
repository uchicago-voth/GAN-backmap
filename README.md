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


### Input file
- A training name must be specified. For this document "INPUTNAME" will be used as an example
- provide all non default settings desired in a file named INPUTNAME.in
- Options are listed below, defaults are in parentheses

### Settings
- NUM_EPOCHS (1000): Number of epochs in training

- TOTAL_SAMPLES (1000): Number of samples in each trajectory

- D_TRAIN_RATE (10): The Generator is only trained every D_TRAIN_RATE batches, to allow for the discriminator to estimate the wasserstein distance before the generator is trained

- BATCH_SIZE (10): Number of samples that the networks train on at one time. Increasing this will speed up training (especially on GPUs) but will reduce regularization and generalization of the networks, often leading to poorer models

- GLEARNING_RATE (0.00008): Learning rate of the generator

- DLEARNING_RATE (0.00005): Learning rate of the discriminator

- BETA1 (0,5): Exponential decay rate for first moment in ADAM optimizer

- Z_SIZE (20): Size of random vector passed to the generator

- G_WIDTH (50): Size of each layer in the generator

- D_WIDTH (80): Size of each layer in the discriminator

- G_DEPTH (4): Number of hidden layers of size G_WIDTH in generator (does not include input and output layers)

- D_DEPTH (8): Number of hidden layers of size D_WIDTH in discriminator (does not include input or output layers)

- CYCLE_LAMBDA (10.0): constant which weights the cyclical loss. Higher values cause training to emphasize cyclical accuracy more

- GP_LAMBDA (10.0): constant which weights the gradient penalty loss (which allows the GAN to use the wasserstein metric properly)

- NUM_DIMS (3): Number of spatial dimensions in the dataset

## 3. Examples
## 4. Dependencies
- Pytorch Version 1.0 or greater
- Numpy
- MDTraj 1.9.3 or greater
- Pandas


## 5. Current Limitations
- Currently only center of mass mappings are accepted
- Only networks with fully connected layers are supported, custom NNs are not available yet


