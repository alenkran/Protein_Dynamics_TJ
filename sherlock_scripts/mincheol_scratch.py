################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Min Cheol's scratch pad python file

################################################################################

# get data
import numpy as np
import argparse as ap

# Set up ArgumentParser
parser = ap.ArgumentParser(description='mincheols scratch pad')
parser.add_argument('-dataset', action='store', dest='which_dataset')
args = parser.parse_args()

# Assign argument variables
which_dataset = args.which_dataset

# scratch pad
X = np.load('/scratch/users/mincheol/' + which_dataset + '/raw_XYZ.dat')
print(X.shape)	