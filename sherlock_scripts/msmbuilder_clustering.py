################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Performs ISOMAP on msm builder tutorial trajectories,
# clusters in reduced dimension, and saves various clustering/dimentionality
# reduction products.

################################################################################

# get data
from msmbuilder.example_datasets import FsPeptide
fs_peptide = FsPeptide()
fs_peptide.cache()

import tempfile
import os
os.chdir(tempfile.mkdtemp())

from msmbuilder.dataset import dataset
xyz = dataset(fs_peptide.data_dir + "/*.xtc",
              topology=fs_peptide.data_dir + '/fs-peptide.pdb',
              stride=10)
print("{} trajectories".format(len(xyz)))
# msmbuilder does not keep track of units! You must keep track of your
# data's timestep
to_ns = 0.5
print("with length {} ns".format(set(len(x)*to_ns for x in xyz)))

#featurization
from msmbuilder.featurizer import DihedralFeaturizer
featurizer = DihedralFeaturizer(types=['phi', 'psi'])
diheds = xyz.fit_transform_with(featurizer, 'diheds/', fmt='dir-npy')

#tICA
from msmbuilder.decomposition import tICA
tica_model = tICA(lag_time=2, n_components=4)
# fit and transform can be done in seperate steps:
tica_model = diheds.fit_with(tica_model)
tica_trajs = diheds.transform_with(tica_model, 'ticas/', fmt='dir-npy')

#histogram

import numpy as np
txx = np.concatenate(tica_trajs)

# clustering
from msmbuilder.cluster import MiniBatchKMeans
clusterer = MiniBatchKMeans(n_clusters=100)
clustered_trajs = tica_trajs.fit_transform_with(
    clusterer, 'kmeans/', fmt='dir-npy'
)

print(tica_trajs[0].shape)
print(clustered_trajs[0].shape)

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
num_clusters=100
cluster_centers = np.empty((num_atoms*num_axis, num_clusters), dtype=float)
X = np.reshape(frames_bag.xyz, (num_frames, num_atoms*num_axis))
for idx in range(num_clusters):
    indices = (cluster_indices == idx)
    cluster_centers[:, idx] = X[indices,:].mean(axis=0)

# save clusters
cluster_centers.dump('/scratch/users/mincheol/isomap_cluster_out/msm_clusters.dat')	

# save assignments
cluster_indices.dump('/scratch/users/mincheol/isomap_cluster_out/msm_clustering_labels.dat')