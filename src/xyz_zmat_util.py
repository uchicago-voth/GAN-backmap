import chemcoord as cc
import fileinput
import time
import numpy as np
import sys
import os.path as path
import pandas as pd
import mdtraj as md

cc.constants.elements['atomic_radius_cc']['C'] =  0.82

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


def to_frag_zmat(name, num_frames, num_atoms):

    num_frames = int(num_frames)
    num_atoms = int(num_atoms)
    infile = name + '.xyz'
    fragfile = name + '.cg'
    resfile = name + '.res'
    outfile = name + '.bad'
    topfile = name + '.zmat'
    frag = np.genfromtxt(fragfile, delimiter = ' ').astype(int)
    resid = open(resfile, 'r').read().splitlines() 
    traj = cc.Cartesian.read_xyz(infile, get_bonds=False)
    print("loaded")
    t = time.time()
    open(outfile, 'w').close()
    out_f = open(outfile, "ab")
    construction_table = make_construction_tables(topfile, frag, resid)
    for i in range(num_frames):
        if i % 500 == 0:
            print("Frame " + str(i))
        frame = traj.iloc[(num_atoms+1) * i : ((num_atoms+1) * i) + num_atoms]
        for j in range(len(frag)-1):
            small_frame = frame.iloc[frag[j] : frag[j+1]]
            small_frame.index = range(frag[j+1] - frag[j])
            small_frame.get_bonds()
            small_frame = center_fragment(small_frame, j, resid)
            
            z_frame = small_frame.get_zmat(construction_table[j])
            #print(z_frame)
            z_frame = z_frame.iloc[:,[2,4,6]]
            np.savetxt(out_f, z_frame.to_numpy(),fmt='%.6f', delimiter=',')
    t_2 = time.time()
    print(t_2 - t)
    out_f.close()



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
    t_2 = time.time()
    print(t_2 - t)

def to_frag_xyz(name, num_frames, num_atoms):
    num_frames = int(num_frames)
    num_atoms = int(num_atoms)
    infile = name + '.bad'
    outfile = name + '_gen.xyz'
    topfile = name + '.zmat'
    cafile = name + '_ca.pdb'
    #catop = name + '_ca.pdb'
    fragfile = name + '.cg'
    resfile = name + '.res'
    ca = md.load(cafile).xyz
    frag = np.genfromtxt(fragfile, delimiter = ' ').astype(int)
    resid = open(resfile, 'r').read().splitlines() 
    z_features = np.genfromtxt(infile, delimiter=',')
    print("loaded")
    t = time.time()
    open(outfile, 'w').close()
    out_f = open(outfile, "a")
    z_mat_list = make_zmats(topfile, frag, resid)
    for i in range(num_frames):
        out_f.write("304\n\n")
        if i % 500 == 0:
            print("Frame " + str(i))
        z_feat_frame = z_features[i*num_atoms:(i+1)*num_atoms, :]
        for j in range(len(frag)-1):
            small_z_feat_frame = z_feat_frame[frag[j] : frag[j+1]]
            z_mat = z_mat_list[j]
            z_mat.unsafe_iloc[:,2] = small_z_feat_frame[:,0]
            z_mat.unsafe_iloc[:,4] = small_z_feat_frame[:,1]
            z_mat.unsafe_iloc[:,6] = small_z_feat_frame[:,2]
            small_xyz_frame = z_mat.get_cartesian()
            coord = np.multiply(10.0, ca[i,j,:])
            #print(coord)
            small_xyz_frame = small_xyz_frame + coord
            small_xyz_frame.to_xyz('temp.xyz')
            temp = open('temp.xyz', 'r')
            out_f.write(''.join(temp.readlines()[2:]))
            out_f.write("\n")
    out_f.close()
            
    t_2 = time.time()
    print(t_2 - t)
    out_f.close()




def center_fragment(small_frame, index, resid):
    #print(small_frame)
    
    if index == 0 or resid[index] == "PRO":
        x = small_frame.iloc[4]['x'].at[4]
        y = small_frame.iloc[4]['y'].at[4]
        z = small_frame.iloc[4]['z'].at[4]
    else:
        x = small_frame.iloc[2]['x'].at[2]
        y = small_frame.iloc[2]['y'].at[2]
        z = small_frame.iloc[2]['z'].at[2]
    small_frame = small_frame - [x,y,z]
    #print(small_frame)
    return small_frame




def make_construction_tables(frame_file, frag, resid):
    construction_table = []
    frame = cc.Cartesian.read_xyz(frame_file)
    for j in range(len(frag)-1):
        small_frame = frame.iloc[frag[j] : frag[j+1]]
        small_frame.index = range(frag[j+1] - frag[j])
        small_frame = center_fragment(small_frame, j, resid)
        bonds = small_frame.get_bonds()
        #print(bonds)
        #print(small_frame)
        if j == 0 or resid[j] == 'PRO':
            c_frag = pd.DataFrame([['origin', 'e_z', 'e_x'], [4, 'e_z', 'e_x'], [4, 0, 'e_x']], columns=['b','a','d'], index = [4, 0, 6])

        #elif j == 1:
        #    c_frag = pd.DataFrame([['origin', 'e_z', 'e_x'], [2, 'e_z', 'e_x'], [2, 0, 'e_x']], columns=['b','a','d'], index = [2, 0, 4])

        else:
            c_frag = pd.DataFrame([['origin', 'e_z', 'e_x'], [2, 'e_z', 'e_x'], [2, 0, 'e_x']], columns=['b','a','d'], index = [2, 0, 4])
        #print(small_frame)
        construction_table.append(small_frame.get_construction_table(fragment_list = [(small_frame, c_frag),]))
        #print(construction_table[j])
    #print(construction_table)
    return construction_table

def make_zmats(frame_file, frag, resid):
    construction_table = []
    frame = cc.Cartesian.read_xyz(frame_file)
    for j in range(len(frag)-1):
        small_frame = frame.iloc[frag[j] : frag[j+1]]
        small_frame.index = range(frag[j+1] - frag[j])
        small_frame = center_fragment(small_frame, j, resid)
        bonds = small_frame.get_bonds()
        #print(bonds)
        #print(small_frame)
        if j == 0 or resid[j] == 'PRO':
            c_frag = pd.DataFrame([['origin', 'e_z', 'e_x'], [4, 'e_z', 'e_x'], [4, 0, 'e_x']], columns=['b','a','d'], index = [4, 0, 6])

        #elif j == 1:
        #    c_frag = pd.DataFrame([['origin', 'e_z', 'e_x'], [2, 'e_z', 'e_x'], [2, 0, 'e_x']], columns=['b','a','d'], index = [2, 0, 4])

        else:
            c_frag = pd.DataFrame([['origin', 'e_z', 'e_x'], [2, 'e_z', 'e_x'], [2, 0, 'e_x']], columns=['b','a','d'], index = [2, 0, 4])
        #print(small_frame)

        zmat = small_frame.get_zmat(construction_table = small_frame.get_construction_table(fragment_list = [(small_frame, c_frag),]))
        construction_table.append(zmat)
        #print(construction_table[j])
    #print(construction_table)
    return construction_table



if len(sys.argv) != 5:
    print("Usage: xyz_zmat_util.py MODE NAME NUM_FRAMES NUM_ATOMS")
    exit()

if sys.argv[1] == 'to_zmat':
    to_zmat(sys.argv[2], sys.argv[3], sys.argv[4])
elif sys.argv[1] == 'to_xyz':
    to_xyz(sys.argv[2], sys.argv[3], sys.argv[4])
elif sys.argv[1] == 'to_frag_zmat':
    to_frag_zmat(sys.argv[2], sys.argv[3], sys.argv[4])
elif sys.argv[1] == 'to_frag_xyz':
    to_frag_xyz(sys.argv[2], sys.argv[3], sys.argv[4])
else:
    print("Mode must either be \'to_xyz\' or \'to_zmat\'")
    exit()
