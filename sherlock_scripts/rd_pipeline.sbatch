#!/bin/bash

#SBATCH --job-name=rd_pipeline
#SBATCH --output=/scratch/users/mincheol/job_outputs/rd_pipeline.%j.out
#SBATCH --error=/scratch/users/mincheol/job_outputs/rd_pipeline.%j.err
#SBATCH --time=12:00:00
#SBATCH --qos=bigmem
#SBATCH -p bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=150GB
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=mincheol@stanford.edu

# otherwise: 
module load anaconda
source activate test_env

python fit_model.py -method isomap -n_neighbors 30 -n_components 10 -dataset fspeptide -stride 15000
python apply_dr.py -method isomap -n_neighbors 30 -n_components 10 -dataset fspeptide -stride 15000
python fit_model.py -method isomap -n_neighbors 30 -n_components 10 -dataset fspeptide -stride 14000
python apply_dr.py -method isomap -n_neighbors 30 -n_components 10 -dataset fspeptide -stride 14000
python apply_dr.py -method isomap -n_neighbors 30 -n_components 10 -dataset fspeptide -stride 13000
python fit_model.py -method isomap -n_neighbors 30 -n_components 10 -dataset fspeptide -stride 12000
python apply_dr.py -method isomap -n_neighbors 30 -n_components 10 -dataset fspeptide -stride 12000
python fit_model.py -method isomap -n_neighbors 30 -n_components 10 -dataset fspeptide -stride 11000
python apply_dr.py -method isomap -n_neighbors 30 -n_components 10 -dataset fspeptide -stride 11000
python fit_model.py -method isomap -n_neighbors 30 -n_components 10 -dataset fspeptide -stride 10000
python apply_dr.py -method isomap -n_neighbors 30 -n_components 10 -dataset fspeptide -stride 10000
python fit_model.py -method isomap -n_neighbors 30 -n_components 10 -dataset fspeptide -stride 9000
python apply_dr.py -method isomap -n_neighbors 30 -n_components 10 -dataset fspeptide -stride 9000

source deactivate
