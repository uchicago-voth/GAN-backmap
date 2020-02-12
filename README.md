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

### Z-matrix utilities

The code is capable of running using cartesian coordinate files or with z-matrices. Z-matrices contain bond, angle, and dihedral information. Here are the steps required to use the z-matrix functionality:
1. First a properly formatted .xyz file must be created for both the AA and CG data. This can be done using VMD, or any other software.
2. The comment lines must be empty in the trajectory, otherwise chemcoord, the package used to convert between xyz and z-matrices, will not function. This can easily be done using sed.
3. Once you have a properly formatted file "INPUTNAME.xyz", use the xyz_zmat_util.py script to generate the internal coordinates of the all atom file..
   - The proper usage of the script is python xyz_zmat_util.py to_zmat INPUTNAME NUM_FRAMES NUM_ATOMS
4. You should now have two files, INPUTNAME.bad and INPUTNAME.zmat. INPUTNAME.bad will contain the Bonds Angles and Dihedrals for each frame in the trajectory, and will be used as the data for training. INPUTNAME.zmat contains a construction table. This is the topological information used to generate the z-matrix, and must be passed as an input for the reverse z-matrix to cartesian transform.
   - The .bad file will be normalized by the code, and should be 3 X NUM_ATOMS dimensional per frame of simlulation, making exactly as complex as the cartesian coordinates. 3N-6 dimensions correspond to internal coordinates, while the remaining 6 dimensions contain rotational and translational information about the structure.
5. Repeat steps 1-4 on the CG trajectory. Bonding topology for this will be generated on the fly. For this step in particular, ensure that the atom names in the .xyz files are atom names that would be recognized by a pdb. The actual identity is unimportant, but chemcoord will throw an error if the atom names in an xyz file are numbers, or if they are residue abbreviations such as 'ALA'. 


**Note: chemcoord was designed to read single configurations at a time, and I have jury-rigged the software to work with whole trajectories. 
The result of this is that there is no way to load trajectories iteratively for conversion to z-matrices. If your trajectory is too large to load into memory all at once,
the code will fail. In such cases, you must split your trajectories into multiple pieces and convert them each separately. Afterwards you can simply cat the .bad files together.
If a .zmat file has been found from a previous usage with the same input name, it will automatically attempt to use it, so that each frame will have the same construction table.** 

After training, the network will output z-matrices, and the to_xyz option of xyz_zmat_util.py can be used (along with the construction table used to generate the z-matrices in the first place) to transform them into cartesian coordinates.  

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
  - Pandas versions have not been extensively tested, but for xyz_zmat utils Pandas 0.25.2 does work, Pandas 1.0.1 does not


## 5. Current Limitations
- Currently only center of mass mappings are accepted
- Only networks with fully connected layers are supported, custom NNs are not available yet



