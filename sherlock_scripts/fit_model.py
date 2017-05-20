################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Fits a dimensionality reduction model and saves the model itself using picke.

# Usage: python fit_model.py -method <method> -n_neighbors <n_neighbors> -n_components <n_components> -dataset <dataset label> -sample_rate <sampling rate>
# method is the method to use. Currently: KernelPCA, Isomap, and SpectralEmbedding
# n_neighbors is a number of neighbors to use
# n_components is components in the ISOMAP space
# dataset is which dataset to use
# sample_rate is the subsampling rate

# Example: python fit_model.py -method isomap -n_neighbors 40 -n_components 10 -dataset fspeptide -stride 60

# Output: 1 pkl file containing the model.
# These files are saved with a *_<n_neighbors>_<n_components>_<sample_rate>.dat naming scheme.

################################################################################

# get data
import numpy as np
import argparse as ap
from sklearn import manifold
from sklearn import decomposition
from sklearn.externals import joblib

# additional imports
import tempfile
import os
os.chdir(tempfile.mkdtemp())

# Set up ArgumentParser
parser = ap.ArgumentParser(description='Fitting model script.')
parser.add_argument('-method', action="store", dest='method')
parser.add_argument('-n_components', action="store", dest='n_components', type=int)
parser.add_argument('-n_neighbors', action="store", dest='n_neighbors', type=int)
parser.add_argument('-dataset', action='store', dest='which_dataset')
parser.add_argument('-stride', action='store', dest='stride', type=int)
args = parser.parse_args()

# Assign argument variables
method = args.method
n_neighbors = args.n_neighbors
n_components = args.n_components
which_dataset = args.which_dataset
stride = args.stride

# Combine n_neighbors and n_components to produce an ID
ID = str(n_neighbors) + '_' + str(n_components) + '_' + str(stride)

# Load appropriate X matrix
X = np.load('/scratch/users/mincheol/' + which_dataset + '/sim_datasets/raw_XYZ_' + str(stride) + '.dat')

# Get dimensions of the X matrix
num_frames = X.shape[0]
num_features = X.shape[1]

# apply dimensionality reduction, fit the model using sample data and transform all other frames as well
if method == 'kernelPCA':
	model = decomposition.KernelPCA(n_components=n_components, kernel='rbf', n_jobs=-1)
elif method == 'isomap':
	model = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, n_jobs=-1)
else: # spectral embedding
	model = manifold.SpectralEmbedding(n_neighbors=n_neighbors, n_components=n_components, n_jobs=-1)

# Fit and save the model
X_iso_sampled = model.fit(X)
joblib.dump(model, '/scratch/users/mincheol/' + which_dataset + '/models/' + method + '_model_' + ID + '.pkl')
print('Model Saved')