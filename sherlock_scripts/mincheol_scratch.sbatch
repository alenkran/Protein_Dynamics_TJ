#!/bin/bash

#SBATCH --job-name=mincheol_scratch
#SBATCH --output=/scratch/users/mincheol/job_outputs/mincheol_scratch.%j.out
#SBATCH --error=/scratch/users/mincheol/job_outputs/mincheol_scratch.%j.err
#SBATCH --time=2:00:00
#SBATCH --qos=dev
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=mincheol@stanford.edu

# otherwise: 
module load anaconda
source activate test_env
python msmbuilder_pipeline.py -num_clusters 100 -dataset fs_peptide

# python perform_PCA.py -method isomap -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 100 -pc 4
# python perform_PCA.py -method isomap -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 150 -pc 4
# python perform_PCA.py -method isomap -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 200 -pc 4
# python perform_PCA.py -method isomap -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 250 -pc 4
# python perform_PCA.py -method isomap -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 300 -pc 4
# python perform_PCA.py -method isomap -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 350 -pc 4
# python perform_PCA.py -method isomap -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 400 -pc 4
# python perform_PCA.py -method isomap -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 450 -pc 4
# python perform_PCA.py -method isomap -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 500 -pc 4
# python perform_PCA.py -method isomap -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 550 -pc 4
# python perform_PCA.py -method isomap -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 600 -pc 4

# python perform_PCA.py -method kernelPCA -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 150 -pc 4
# python perform_PCA.py -method kernelPCA -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 200 -pc 4
# python perform_PCA.py -method kernelPCA -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 250 -pc 4
# python perform_PCA.py -method kernelPCA -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 300 -pc 4
# python perform_PCA.py -method kernelPCA -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 350 -pc 4
# python perform_PCA.py -method kernelPCA -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 400 -pc 4
# python perform_PCA.py -method kernelPCA -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 450 -pc 4
# python perform_PCA.py -method kernelPCA -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 500 -pc 4
# python perform_PCA.py -method kernelPCA -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 550 -pc 4
# python perform_PCA.py -method kernelPCA -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 600 -pc 4

# python perform_PCA.py -method kernelPCA -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 60 -pc 4

# python generate_subsamples.py
source deactivate
