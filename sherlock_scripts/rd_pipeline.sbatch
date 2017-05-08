#!/bin/bash

#SBATCH --job-name=rd_pipeline
#SBATCH --output=/scratch/users/mincheol/job_outputs/rd_pipeline.%j.out
#SBATCH --error=/scratch/users/mincheol/job_outputs/rd_pipeline.%j.err
#SBATCH --time=12:00:00
#SBATCH --qos=bigmem
#SBATCH -p bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=256GB
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=mincheol@stanford.edu

# otherwise: 
module load anaconda
source activate test_env
python fit_model.py -method isomap -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 50
python apply_dr.py -method isomap -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 50
source deactivate
