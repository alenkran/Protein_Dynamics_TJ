################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Takes in ISOMAP dimensionality reduction / clustering results and computes
# the distance matrices, both for frames and clusters. 

# Usage: python isomap_compute_distances -n_neighbors <n_neighbors> -n_components <n_components> -num_clusters <num_clusters>
# n_neighbors is a number of neighbors used for kernel PCA
# n_components is components used in the ISOMAP space
# num_clusters is the number of clusters used for clustering

# Example: python isomap_compute_distances.py -n_neighbors 30 -n_components 40 -num_clusters 97

# Output: 2 files
# 1) Distance matrix (num_frames x num_frames) between each and every frame (in ISOMAP dimension)
# 2) Distance matrix (num_states x num_states) between each and every state (in ISOMAP dimension)

################################################################################

# Imports
import numpy as np
import argparse as ap

# Set up ArgumentParser
parser = ap.ArgumentParser(description='ISOMAP processing script.')
parser.add_argument('-n_components', action="store", dest='n_components', type=int)
parser.add_argument('-n_neighbors', action="store", dest='n_neighbors', type=int)
parser.add_argument('-num_clusters', action="store", dest='num_clusters', type=int, default=97)
args = parser.parse_args()

# Assign argument variables
n_neighbors = args.n_neighbors
n_components = args.n_components
num_clusters = args.num_clusters

# Combine n_neighbors and n_components to produce an ID
ID = str(n_neighbors) + '_' + str(n_components) + '_' + str(num_clusters)

# Load dimensionality and clustering results
cluster_centers_XYZ = np.load('/scratch/users/mincheol/isomap_cluster_out/isomap_clusters_XYZ_' + ID + '.dat')	
cluster_centers_RD = np.load('/scratch/users/mincheol/isomap_cluster_out/isomap_clusters_RD_' + ID + '.dat')
cluster_labels = np.load('/scratch/users/mincheol/isomap_cluster_out/isomap_clustering_labels_' + ID + '.dat')
X_iso = np.load('/scratch/users/mincheol/isomap_cluster_out/isomap_coordinates_' + ID + '.dat')
print('All data loaded')


# Compute distance matrix from and to frame in ISOMAP dimension
num_frames = X_iso.shape[0]
frame_distance_matrix = np.zeros((num_frames, num_frames))
for frame1 in range(num_frames):
    for frame2 in range(frame1, num_frames):
        frame_distance_matrix[frame1][frame2] = np.linalg.norm(X_iso[frame1, :] - X_iso[frame2,:])
frame_distance_matrix = frame_distance_matrix + frame_distance_matrix.transpose()
print('Frame distances computed')

# Compute distance matrix between cluster centers in ISOMAP dimension
num_states = cluster_centers_RD.shape[0]
state_distance_matrix = np.zeros((num_states, num_states))
for state1 in range(num_states):
    for state2 in range(state1, num_states):
        state_distance_matrix[state1][state2] = np.linalg.norm(cluster_centers_RD[state1,:] - cluster_centers_RD[state2,:])
state_distance_matrix = state_distance_matrix + state_distance_matrix.transpose()
print('State distances computed')

# Save the frame distance matrix
frame_distance_matrix.dump('/scratch/users/mincheol/isomap_cluster_out/frame_dist_mat_' + ID + '.dat')	
print('Frame distances saved')

# Save the state distances matrix
state_distance_matrix.dump('/scratch/users/mincheol/isomap_cluster_out/state_dist_mat_' + ID + '.dat')
print('State distances saved')
