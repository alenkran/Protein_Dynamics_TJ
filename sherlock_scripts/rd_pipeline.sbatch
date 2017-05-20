#!/bin/bash

#SBATCH --job-name=rd_pipeline
#SBATCH --output=/scratch/users/mincheol/job_outputs/rd_pipeline.%j.out
#SBATCH --error=/scratch/users/mincheol/job_outputs/rd_pipeline.%j.err
#SBATCH --time=4:00:00
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --mem=60GB
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=mincheol@stanford.edu

# otherwise: 
module load anaconda
source activate test_env

python fit_model.py -method kernelPCA -n_neighbors 30 -n_components 10 -dataset apo_calmodulin -stride 1000
python apply_dr.py -method kernelPCA -n_neighbors 30 -n_components 10 -dataset apo_calmodulin -stride 1000

source deactivate
