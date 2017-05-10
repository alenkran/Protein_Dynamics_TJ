#!/bin/bash

#SBATCH --job-name=rd_pipeline
#SBATCH --output=/scratch/users/mincheol/job_outputs/rd_pipeline.%j.out
#SBATCH --error=/scratch/users/mincheol/job_outputs/rd_pipeline.%j.err
#SBATCH --time=12:00:00
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
python apply_dr.py -method isomap -n_neighbors 40 -n_components 10 -dataset holo_calmodulin -stride 100
python apply_dr.py -method isomap -n_neighbors 40 -n_components 10 -dataset holo_calmodulin -stride 150
python apply_dr.py -method isomap -n_neighbors 40 -n_components 10 -dataset holo_calmodulin -stride 200
python apply_dr.py -method isomap -n_neighbors 40 -n_components 10 -dataset holo_calmodulin -stride 250
python apply_dr.py -method isomap -n_neighbors 40 -n_components 10 -dataset holo_calmodulin -stride 300
python apply_dr.py -method isomap -n_neighbors 40 -n_components 10 -dataset holo_calmodulin -stride 350
python apply_dr.py -method isomap -n_neighbors 40 -n_components 10 -dataset holo_calmodulin -stride 400
python apply_dr.py -method isomap -n_neighbors 40 -n_components 10 -dataset holo_calmodulin -stride 450
python apply_dr.py -method isomap -n_neighbors 40 -n_components 10 -dataset holo_calmodulin -stride 500
python apply_dr.py -method isomap -n_neighbors 40 -n_components 10 -dataset holo_calmodulin -stride 550
python apply_dr.py -method isomap -n_neighbors 40 -n_components 10 -dataset holo_calmodulin -stride 600
source deactivate
