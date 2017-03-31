################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Run msmbuilder on the bootstrap data and save results

# Usage: python msmbuilder_compare.py -dataset <dataset> -num_clusters <num_clusters>
# dataset is which protein dataset to examine
# num_clusters is the number of clusters used for clustering


# Example: python msmbuilder_bootstrap.py -dataset fspeptide -num_clusters 97

# Outputs: eigenvectors and eigenvalues

################################################################################

# imports / get data
from msmbuilder.example_datasets import FsPeptide
import numpy as np
import scipy as scipy
import argparse as ap
from msmbuilder.dataset import dataset
import mdtraj as md
import tempfile
import os
import random

# Process arguments
parser = ap.ArgumentParser(description='MSMBuilder comparions.')
parser.add_argument('-dataset', action='store', dest='which_dataset')
parser.add_argument('-num_clusters', action='store', dest='num_clusters', type=int, default=97)
args = parser.parse_args()

# Assignment arguments
which_dataset = args.which_dataset
num_clusters = args.num_clusters

# Read original data
os.chdir(tempfile.mkdtemp())

xyz = [] # placeholder
if which_dataset == 'fspeptide':
    # Get data
    fs_peptide = FsPeptide()
    fs_peptide.cache()
    data_dir = '/scratch/users/mincheol/' + which_dataset + '/trajectories/temp'
    xyz = dataset(data_dir + "/*.xtc",
                  topology=fs_peptide.data_dir + '/fs-peptide.pdb',
                  stride=10)
    print("{} trajectories".format(len(xyz)))
    # msmbuilder does not keep track of units! You must keep track of your
    # data's timestep
    to_ns = 0.5
    print("with length {} ns".format(set(len(x)*to_ns for x in xyz)))

if which_dataset == 'calmodulin':
    print('No data here...')

#featurization
from msmbuilder.featurizer import DihedralFeaturizer
featurizer = DihedralFeaturizer(types=['phi', 'psi'])
diheds = xyz.fit_transform_with(featurizer, 'diheds/', fmt='dir-npy')

#tICA
from msmbuilder.decomposition import tICA
random.seed(0)
tica_model = tICA(lag_time=2, n_components=4)
# fit and transform can be done in seperate steps:
tica_model = diheds.fit_with(tica_model)
tica_trajs = diheds.transform_with(tica_model, 'ticas/', fmt='dir-npy')

txx = np.concatenate(tica_trajs)

# clustering
from msmbuilder.cluster import MiniBatchKMeans
random.seed(0)
clusterer = MiniBatchKMeans(n_clusters=num_clusters)
clustered_trajs = tica_trajs.fit_transform_with(
    clusterer, 'kmeans/', fmt='dir-npy'
)

# msm builder
from msmbuilder.msm import MarkovStateModel
from msmbuilder.utils import dump
msm = MarkovStateModel(lag_time=2, n_timescales=20, ergodic_cutoff='off')
msm.fit(clustered_trajs)

# save tIC plot
import matplotlib
matplotlib.use('Agg') # Must be placed before matplotlib.pyplot
import matplotlib.pyplot as plt
plt.hexbin(txx[:, 0], txx[:, 1], bins='log', mincnt=1, cmap="bone_r")
plt.scatter(clusterer.cluster_centers_[msm.state_labels_, 0],
            clusterer.cluster_centers_[msm.state_labels_, 1],
            s=1e4 * msm.populations_,       # size by population
            c=msm.left_eigenvectors_[:, 1], # color by eigenvector
            cmap="coolwarm") 
plt.colorbar(label='First dynamical eigenvector')
plt.xlabel('tIC 1')
plt.ylabel('tIC 2')

num_traj = str(len(xyz))
traj_length = str(len(xyz[0].xyz))
plot_title = 'num_traj: ' + num_traj + 'traj_lenth: ' + traj_length
plt.title(plot_title)
plt.tight_layout()
filename = num_traj + '_' + traj_length
plt.savefig('/scratch/users/mincheol/' + which_dataset + '/trajectories/' + filename + '.png') #
print('tIC plot saved')

# Saved the new eigenvectors
print('Bootstrap values:')
print(msm.eigenvalues_)
v = msm.eigenvalues_
lv = msm.left_eigenvectors_

# Sort eigenvectors  and eigenvalues
idx_sort = msm.populations_.argsort()[-len(msm.populations_):][::-1]
lv = lv[idx_sort,:]

# Save the eigenvectors and eigvenvalues
print(filename)
v.dump('/scratch/users/mincheol/' + which_dataset + '/trajectories/' + 'v_' + filename + '.dat')
lv.dump('/scratch/users/mincheol/' + which_dataset + '/trajectories/' + 'lv_' + filename + '.dat')
print('\ndone\n')