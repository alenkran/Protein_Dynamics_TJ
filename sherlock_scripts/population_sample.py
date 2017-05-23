################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Samples the population of MSM builder.

# Usage: python sample_msm.py -num_clusters <num_clusters>
# num_clusters is the number of clusters used for clustering

# Example: python population_sample.py -num_clusters 400 -dataset apo_calmodulin

# Outputs: 3 files
# 1) raw_XYZ - trajectory in raw format (each row is a frame)
# 3) cluster_indices - state assignment matrix for each frame in raw_XYZ

################################################################################

# imports / get data
from msmbuilder.example_datasets import FsPeptide
import numpy as np
import scipy as scipy
import argparse as ap
from msmbuilder.dataset import dataset
import mdtraj as md

# Process arguments
parser = ap.ArgumentParser(description='MSMBuilder pipeline script.')
parser.add_argument('-num_clusters', action='store', dest='num_clusters', type=int)
parser.add_argument('-dataset', action='store', dest='which_dataset')
args = parser.parse_args()

# Assignment arguments
num_clusters = args.num_clusters
which_dataset = args.which_dataset

import tempfile
import os
os.chdir(tempfile.mkdtemp())

xyz = [] # placeholder
if which_dataset == 'fspeptide':
	# Get data
	fs_peptide = FsPeptide()
	fs_peptide.cache()
	xyz = dataset(fs_peptide.data_dir + "/*.xtc",
	              topology=fs_peptide.data_dir + '/fs-peptide.pdb',
	              stride=10)
	print("{} trjaectories".format(len(xyz)))
	# msmbuilder does not keep track of units! You must keep track of your
	# data's timestep
	to_ns = 0.5
	print("with length {} ns".format(set(len(x)*to_ns for x in xyz)))

if which_dataset == 'apo_calmodulin':
	print('correct')
	xyz = dataset('/scratch/users/mincheol/apo_trajectories' + '/*.lh5', stride=1)

#featurization
from msmbuilder.featurizer import DihedralFeaturizer
featurizer = DihedralFeaturizer(types=['phi', 'psi'])
print(xyz)
print(which_dataset)
diheds = xyz.fit_transform_with(featurizer, 'diheds/', fmt='dir-npy')  #?????????????????????????????

#tICA
from msmbuilder.decomposition import tICA

if which_dataset == 'fspeptide':
	tica_model = tICA(lag_time=2, n_components=4)
if which_dataset == 'apo_calmodulin':
	tica_model = tICA(lag_time=250, n_components=20)

# fit and transform can be done in seperate steps:
tica_model = diheds.fit_with(tica_model)
tica_trajs = diheds.transform_with(tica_model, 'ticas/', fmt='dir-npy')

txx = np.concatenate(tica_trajs)

# clustering
from msmbuilder.cluster import MiniBatchKMeans
clusterer = MiniBatchKMeans(n_clusters=num_clusters) #100 for camodulin
clustered_trajs = tica_trajs.fit_transform_with(
    clusterer, 'kmeans/', fmt='dir-npy'
)

# msm builder
from msmbuilder.msm import MarkovStateModel
from msmbuilder.utils import dump

if which_dataset == 'fspeptide':
	msm = MarkovStateModel(lag_time=2, n_timescales=20, ergodic_cutoff='on')
if which_dataset == 'apo_calmodulin':
	msm = MarkovStateModel(lag_time=125, n_timescales=20, ergodic_cutoff='on')

msm.fit(clustered_trajs)

# combine all trajectories into a trajectory "bag"
frames_bag = []
for idx, trajectories in enumerate(xyz):
    if idx == 0:
        frames_bag = trajectories
    if idx != 0:
        frames_bag = frames_bag.join(trajectories)
num_frames, num_atoms, num_axis = frames_bag.xyz.shape

# Concatenate the trajectories in cluster indices
cluster_indices = np.concatenate(clustered_trajs)

# compute XYZ coordinates of cluster centers
cluster_centers = np.empty((num_clusters, num_atoms*num_axis), dtype=float)
X = np.reshape(frames_bag.xyz, (num_frames, num_atoms*num_axis))
for idx in range(num_clusters):
    indices = (cluster_indices == idx)
    cluster_centers[idx, :] = X[indices,:].mean(axis=0)

folder = '/scratch/users/mincheol/' + which_dataset + '/sim_datasets/'

# save MSM
import msmbuilder.utils as msmutils
msmutils.dump(msm, '/scratch/users/mincheol/' + which_dataset + '/sim_datasets/msm.pkl')

# save clusters
np.savetxt(folder + 'msm_clusters_XYZ.csv', cluster_centers, delimiter=',')

# save assignments
np.savetxt(folder + 'msm_clustering_labels.csv', cluster_indices, delimiter=',')

# save population vector
np.savetxt(folder + 'msm_pop_vec.csv', msm.populations_, delimiter=',')

# Generate sampled data
# Find frame limit

limit_list = []
state_list = []
for state in msm.mapping_.keys():
    num_frame = np.where(cluster_indices == state)[0].shape[0]
    prob = msm.populations_[msm.mapping_[state]]
    limit = num_frame/prob
    limit_list.append(limit)
    state_list.append(state)

limiting_state = state_list[np.argmin(limit_list)] #original frame label
max_frame = int(limit_list[msm.mapping_[limiting_state]])


for num_frame in np.arange(5000, max_frame, 1000):

	# Number of frames to sample from each state
	num_state_frames = np.array(num_frame*msm.populations_).astype(int)

	# Go through each state and take the appropriate number of frames
	frame_idx = np.empty((0,))
	for state in msm.mapping_.keys():
	    options = np.where(cluster_label == state)[0]
	    frame_idx = np.hstack((frame_idx,np.random.choice(options, num_state_frames[msm.mapping_[state]], replace=False)))
	frame_idx = frame_idx.astype(int)

	# Save data
	X_hat = X[frame_idx, :]
	np.savetxt(folder + 'raw_XYZ_'+str(num_frame)+'.csv', X_hat, delimiter=',')
	np.savetxt(folder + 'indices_'+str(num_frame)+'.csv', frame_idx, delimiter=',')
