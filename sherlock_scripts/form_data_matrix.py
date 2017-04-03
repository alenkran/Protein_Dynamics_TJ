################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Generates the raw X matrix (rows are frames, cols are XYZ-XYZ-... coordinates)

# Usage: python form_data_matrix.py -dataset <dataset label>
# dataset label is which dataset to use

# Example: python form_data_matrix.py -dataset calmodulin

# Output: 1 file
# 1) X matrix, matrix form of the raw frames (frame bag)


################################################################################

# get data
import numpy as np
import argparse as ap
import mdtraj as md
from sklearn import manifold
from msmbuilder.example_datasets import FsPeptide
import sys
fs_peptide = FsPeptide()
fs_peptide.cache()

# additional imports
import tempfile
import os
os.chdir(tempfile.mkdtemp())

# Set up ArgumentParser
parser = ap.ArgumentParser(description='X matrix forming script.')
parser.add_argument('-dataset', action='store', dest='which_dataset')
args = parser.parse_args()

# Assign argument variables
which_dataset = args.which_dataset

# Compile all trajectories
from msmbuilder.dataset import dataset
xyz = [] # placeholder
if which_dataset == 'fspeptide':
	# Get data
	fs_peptide = FsPeptide()
	fs_peptide.cache()
	xyz = dataset(fs_peptide.data_dir + "/*.xtc",
	              topology=fs_peptide.data_dir + '/fs-peptide.pdb',
	              stride=10)
	print("{} trajectories".format(len(xyz)))
	# msmbuilder does not keep track of units! You must keep track of your
	# data's timestep
	to_ns = 0.5
	print("with length {} ns".format(set(len(x)*to_ns for x in xyz)))

if which_dataset == 'calmodulin':
	xyz = dataset('/scratch/users/mincheol/Trajectories' + '/*.lh5')
	print("{} trajectories".format(len(xyz)))

# Combine all trajectories into a trajectory "bag"
temp = xyz[0]
_, num_atoms, num_axis = temp.xyz.shape
reference_frame = temp.slice(0, copy=True)
num_features = num_atoms*num_axis;
pre_X = [np.reshape(traj.xyz, (traj.superpose(reference_frame).xyz.shape[0],num_features)) for traj in xyz]
X = np.concatenate(pre_X)
X.dump('/scratch/users/mincheol/' + which_dataset + '/raw_XYZ.dat')	