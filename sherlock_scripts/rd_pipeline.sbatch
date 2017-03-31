#!/bin/bash

#SBATCH --job-name=isomap_cluster
#SBATCH --output=/scratch/users/mincheol/job_outputs/rd_pipeline.%j.out
#SBATCH --error=/scratch/users/mincheol/job_outputs/rd_pipeline.%j.err
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=60GB
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=mincheol@stanford.edu

# otherwise: 
module load anaconda
source activate test_env
python fit_model.py -method isomap -n_neighbors 40 -n_components 10 -dataset fspeptide -sample_rate 0.1
#python apply_dr.py -method isomap -n_neighbors 30 -n_components 2 -dataset fspeptide -sample_rate 1.0
source deactivate
