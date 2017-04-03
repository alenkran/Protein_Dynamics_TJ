################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Applies a dimensionality reduction on a given dataset. 

# Usage: python isomap_clustering.py -method <method> -n_neighbors <n_neighbors> -n_components <n_components> -dataset <dataset label>
# method is the method to use. Currently: KernelPCA, Isomap, and SpectralEmbedding
# n_neighbors is a number of neighbors to use
# n_components is components in the ISOMAP space
# dataset is which dataset to use

# Example: python apply_dr.py -method isomap -n_neighbors 100 -n_components 100 -dataset apo_calmodulin -sample_rate 0.1

# Output: 1 .dat file containing the transformed coordinates.
# These files are saved with a *_<n_neighbors>_<n_components>_<sample_rate>.dat naming scheme.

################################################################################

# imports
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
parser.add_argument('-sample_rate', action='store', dest='sample_rate', type=float)
args = parser.parse_args()

# Assign argument variables
method = args.method
n_neighbors = args.n_neighbors
n_components = args.n_components
which_dataset = args.which_dataset
sample_rate = args.sample_rate

# Combine n_neighbors and n_components to produce an ID
ID = str(n_neighbors) + '_' + str(n_components) + '_' + str(sample_rate)

# Load appropriate X matrix
X = np.load('/scratch/users/mincheol/' + which_dataset + '/raw_XYZ.dat')
print('Raw data loaded')
# Load the appropriate model 
model = joblib.load('/scratch/users/mincheol/' + which_dataset + '/models/' + method + '_model_' + ID + '.pkl')
print('Model loaded')
# Transform X matrix in batches
X_rd = np.empty((X.shape[0], n_components))
num_batches = 100
batch_size = int(X.shape[0]/num_batches) # size of each batch
for batch in range(num_batches+1):
	start_idx = batch * batch_size
	end_idx = (batch + 1)*batch_size if (batch + 1)*batch_size < X.shape[0] else X.shape[0]
	if start_idx != end_idx:
		X_rd[start_idx:end_idx, :] = model.transform(X[start_idx:end_idx,:])

# Saved X in reduced dimension
X_rd.dump('/scratch/users/mincheol/' + which_dataset + '/reduced_dimension/X_'+ method +'_' + ID + '.dat')
print('Coordinates saved in reduced dimension')

