################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Performs ISOMAP on msm builder tutorial trajectories,
# clusters in reduced dimension, and saves various clustering/dimentionality
# reduction products.

# Usage: python isomap_clustering.py -n_neighbors <n_neighbors> -n_components <n_components> -num_clusters <num_clusters> -dataset <dataset label> -sample_rate <sampling rate>
# n_neighbors is a number of neighbors to use for kernel PCA
# n_components is components in the ISOMAP space
# num_clusters is the number of clusters used for clustering
# dataset is which dataset to use
# sample_rate is the subsampling rate

# Example: python isomap_clustering.py -n_neighbors 1000 -n_components 1000 -num_clusters 200 -dataset calmodulin -sample_rate 0.1

# Output: 3 files
# 1) Cluster centers in XYZ dimension
# 2) Cluster centers in reduced dimension
# 3) Cluster assignments
# 4) ISOMAP coordinates of raw frames
# These files are saved with a *_<n_neighbors>_<n_components>_<num_clusters>_.dat naming scheme.


################################################################################

# get data
import numpy as np
import argparse as ap
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
parser = ap.ArgumentParser(description='ISOMAP processing script.')
parser.add_argument('-n_components', action="store", dest='n_components', type=int)
parser.add_argument('-n_neighbors', action="store", dest='n_neighbors', type=int)
parser.add_argument('-num_clusters', action="store", dest='num_clusters', type=int, default=97)
parser.add_argument('-dataset', action='store', dest='which_dataset')
parser.add_argument('-sample_rate', action='store', dest='sample_rate', type=float)
args = parser.parse_args()

# Assign argument variables
n_neighbors = args.n_neighbors
n_components = args.n_components
num_clusters = args.num_clusters
which_dataset = args.which_dataset
sample_rate = args.sample_rate

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
	xyz = dataset('/scratch/users/mincheol/Trajectories' + '/*.lh5', stride=10)
	print("{} trajectories".format(len(xyz)))

# Combine all trajectories into a trajectory "bag"
import mdtraj as md

temp = xyz[0]
_, num_atoms, num_axis = temp.xyz.shape
reference_frame = temp.slice(0, copy=True)
num_features = num_atoms*num_axis;
pre_X = [np.reshape(traj.xyz, (traj.superpose(reference_frame).xyz.shape[0],num_features)) for traj in xyz]
X = np.concatenate(pre_X)
X.dump('/scratch/users/mincheol/' + which_dataset + '/raw_XYZ.dat')	

# Subsample
desired_num_frames = int(round(sample_rate*num_frames))
indices = [i for i in range(num_frames)]
np.random.shuffle(indices)
indices = indices[:desired_num_frames]
X = X[indices,:]

#apply dimensionality reduction
X_iso = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, n_jobs=32).fit_transform(X)
print("Sent to ISOMAP land")

X_iso.dump('/scratch/users/mincheol/' + which_dataset + '/isomap_out/isomap_coordinates_' + ID + '.dat')
print("Isomap coordinates of raw frames saved")

# use K means to cluster and save data
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=num_clusters).fit(X_iso)
print("Clustered in ISOMAP space")

# compute XYZ coordinates of cluster centers
cluster_centers = np.empty((num_clusters, num_atoms*num_axis), dtype=float)
for idx in range(num_clusters):
    indices = (kmeans.labels_ == idx)
    cluster_centers[idx, :] = X[indices,:].mean(axis=0)

# Combine n_neighbors and n_components to produce an ID
ID = str(n_neighbors) + '_' + str(n_components) + '_' + str(num_clusters) + '_' + str(sample_rate)

# save centroids in XYZ space
cluster_centers.dump('/scratch/users/mincheol/' + which_dataset + '/isomap_out/isomap_clusters_XYZ_' + ID + '.dat')	
print("Cluster centers saved in XYZ coordinates")

# save centroids in ISOMAP space
kmeans.cluster_centers_.dump('/scratch/users/mincheol/' + which_dataset + '/isomap_out/isomap_clusters_RD_' + ID + '.dat')
print("Cluster centers saved in reduced dimension")

# save assignments
kmeans.labels_.dump('/scratch/users/mincheol/' + which_dataset + '/isomap_out/isomap_clustering_labels_' + ID + '.dat')
print("Clusters assignments saved")

# save the isomap coordinates


