#!/bin/bash

#SBATCH --job-name=mincheol_scratch
#SBATCH --output=/scratch/users/mincheol/job_outputs/mincheol_scratch.%j.out
#SBATCH --error=/scratch/users/mincheol/job_outputs/mincheol_scratch.%j.err
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=60GB
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=mincheol@stanford.edu

# otherwise: 
module load anaconda
source activate test_env
# python form_data_matrix.py -dataset apo_calmodulin -stride 70
# python form_data_matrix.py -dataset apo_calmodulin -stride 80
# python form_data_matrix.py -dataset apo_calmodulin -stride 90
# python form_data_matrix.py -dataset apo_calmodulin -stride 100
# python form_data_matrix.py -dataset apo_calmodulin -stride 110
# python form_data_matrix.py -dataset apo_calmodulin -stride 150
# python form_data_matrix.py -dataset apo_calmodulin -stride 200
# python form_data_matrix.py -dataset apo_calmodulin -stride 250

python perform_PCA.py -method isomap -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 150 -pc 4
python perform_PCA.py -method isomap -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 200 -pc 4
python perform_PCA.py -method isomap -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 250 -pc 4

python perform_PCA.py -method kernelPCA -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 60 -pc 4
source deactivate
