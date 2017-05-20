################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Min Cheol's scratch pad python file

################################################################################

# get data
import numpy as np
from sklearn.decomposition import PCA

# for i in range(25, 46):
# 	stride = i*10
# 	filename = '/scratch/users/mincheol/apo_calmodulin/datasets/indices_' + str(stride) + '.dat'
# 	indices = np.load(filename)
# 	print(stride)
# 	print(indices.shape)

n_neighbors_set = [30]
n_components = 10
strides = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
which_dataset = 'apo_calmodulin'
methods = ['isomap', 'kernelPCA']
pc = 4

for n_neighbors in n_neighbors_set:
	for stride in strides:
		for method in methods:
			print (n_neighbors,stride,method)
			ID = str(n_neighbors) + '_' + str(n_components) + '_' + str(stride)

			# Load things
			X_rd = np.load('/scratch/users/mincheol/' + which_dataset + '/reduced_dimension/X_'+ method +'_' + ID + '.dat')

			# Remove NaNs
			indices = []
			for i in range(X_rd.shape[0]):
			    if np.isnan(np.sum(X_rd[i,:])) or np.isinf(np.sum(X_rd[i,:])) or np.sum(X_rd[i,:]) > 150 or np.sum(X_rd[i,:]) < -150:
			        indices.append(i)

			X_rd = np.delete(X_rd, indices, 0)

			# Perform PCA
			pca = PCA(n_components=pc)
			X_rp = pca.fit_transform(X_rd)

			# Save the PC coordinates
			X_rp.dump('/scratch/users/mincheol/' + which_dataset + '/principal_components/X_pc_'+ method +'_' + ID + '_' + str(pc) + '.dat')