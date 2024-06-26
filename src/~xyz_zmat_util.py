import chemcoord as cc
import fileinput
import time
import numpy as np
import sys
import os.path as path

def to_zmat(name, num_frames, num_atoms):
    num_frames = int(num_frames)
    num_atoms = int(num_atoms)
    infile = name + '.xyz'
    outfile = name + '.bad'
    topfile = name + '.zmat'
    traj = cc.Cartesian.read_xyz(infile, get_bonds=False)
    print("loaded")
    t = time.time()
    open(outfile, 'w').close()
    out_f = open(outfile, "ab")
    construction_table = 0
    for i in range(num_frames):
        if i % 500 == 0:
            print("Frame " + str(i))
        #frame = cc.Cartesian.read_xyz(infile, start_index=1, nrows = num_atoms)
        #print(frame)
        frame = traj.iloc[(num_atoms+1) * i : ((num_atoms+1) * i) + num_atoms]
        frame.get_bonds()
        frame.index = range(num_atoms)
        if i == 0:
            if path.exists(topfile):
                print('Reading previously generated construction table')
                top = cc.Zmat.read_zmat(topfile, implicit_index=False)
                construction_table = top.get_cartesian().get_construction_table()
            else:
                construction_table = frame.get_construction_table()
        z_frame = frame.get_zmat(construction_table)
        #sorted_dataframe = z_frame.sort_index()
        #z_frame.unsafe_iloc[:,:] = sorted_dataframe.to_numpy()[:,:]
        #print(z_frame)
        if i == 0:
            if not path.exists(topfile):
                top = z_frame
                top.to_zmat(topfile, implicit_index=False)
        z_frame = z_frame.iloc[:,[2,4,6]]
        np.savetxt(out_f, z_frame.to_numpy(),fmt='%.6f', delimiter=',')
    t_2 = time.time()
    print(t_2 - t)
    out_f.close()


#def to_frag_zmat(name, num_frames, num_atoms):
#    num_frames = int(num_frames)
#    num_atoms = int(num_atoms)
#    infile = name + '.xyz'
#    fragfile = name + '.cg'
#    outfile = name + '.bad'
#    topfile = name + '.zmat'
#    frag = np.genfromtxt(fragfile, delimiter=' ')
#    traj = cc.Cartesian.read_xyz(infile, get_bonds=False)
#    print("loaded")
#    t = time.time()
#    open(outfile, 'w').close()
#    out_f = open(outfile, "ab")
#    construction_table = []
#    for i in range(num_frames):
#        if i % 500 == 0:
#            print("Frame " + str(i))
#        frame = traj.iloc[(num_atoms+1) * i : ((num_atoms+1) * i) + num_atoms]
#        for i in range(len(frag.shape[0])):
#            small_frame = frame.iloc[
#        #frame = cc.Cartesian.read_xyz(infile, start_index=1, nrows = num_atoms)
#        #print(frame)
#        frame.get_bonds()
#        frame.index = range(num_atoms)
#        if i == 0:
#            if path.exists(topfile):
#                print('Reading previously generated construction table')
#                top = cc.Zmat.read_zmat(topfile, implicit_index=False)
#                construction_table = top.get_cartesian().get_construction_table()
#            else:
#                construction_table = frame.get_construction_table()
#        z_frame = frame.get_zmat(construction_table)
#        #sorted_dataframe = z_frame.sort_index()
#        #z_frame.unsafe_iloc[:,:] = sorted_dataframe.to_numpy()[:,:]
#        #print(z_frame)
#        if i == 0:
#            if not path.exists(topfile):
#                top = z_frame
#                top.to_zmat(topfile, implicit_index=False)
#        z_frame = z_frame.iloc[:,[2,4,6]]
#        np.savetxt(out_f, z_frame.to_numpy(),fmt='%.6f', delimiter=',')
#    t_2 = time.time()
#    print(t_2 - t)
#    out_f.close()



def to_xyz(name, num_frames, num_atoms):
    num_frames = int(num_frames)
    num_atoms = int(num_atoms)
    infile = name + '.bad'
    outfile = name + '_gen.xyz'
    topfile = name + '.zmat'
    top = cc.Zmat.read_zmat(topfile, implicit_index=False)
    open(outfile, 'w').close()
    out_f = open(outfile, 'a')
    t = time.time()
    z_features = np.genfromtxt(infile, delimiter=',')
    for i in range(num_frames):
        if i % 500 == 0:
            print("Frame " + str(i))
        z_feat_frame = z_features[i*num_atoms:(i+1)*num_atoms,:]
        z_mat = top
        z_mat.unsafe_iloc[:,2] = z_feat_frame[:,0]
        z_mat.unsafe_iloc[:,4] = z_feat_frame[:,1]
        z_mat.unsafe_iloc[:,6] = z_feat_frame[:,2]
        xyz_frame = z_mat.get_cartesian()
        xyz_frame.to_xyz('temp.xyz')
        temp = open('temp.xyz', 'r')
        out_f.write(temp.read())
        out_f.write("\n")
    out_f.close()
    open('single.xyz', 'w').close()
    t_2 = time.time()
    print(t_2 - t)

if len(sys.argv) != 5:
    print("Usage: xyz_zmat_util.py MODE NAME NUM_FRAMES NUM_ATOMS")
    exit()

if sys.argv[1] == 'to_zmat':
    to_zmat(sys.argv[2], sys.argv[3], sys.argv[4])
elif sys.argv[1] == 'to_xyz':
    to_xyz(sys.argv[2], sys.argv[3], sys.argv[4])
else:
    print("Mode must either be \'to_xyz\' or \'to_zmat\'")
    exit()
