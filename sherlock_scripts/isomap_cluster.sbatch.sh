#!/bin/bash 
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
#set a job name  
#SBATCH --job-name=isomap_cluster
#################  
#a file for job output, you can check job progress, append the job ID with %j to make it unique
#SBATCH --output=isomap_cluster_output.%j.out

#################
# a file for errors from the job
#SBATCH --error=isomap_cluster_output.%j.err

#################
#time you think you need; default is 2 hours
#format could be dd-hh:mm:ss, hh:mm:ss, mm:ss, or mm
#SBATCH --time=5:00:00

#################
#Quality of Service (QOS); think of it as job priority, there is also --qos=long for with a max job length of 7 days, qos normal is 48 hours.
# REMOVE "normal" and set to "long" if you want your job to run longer than 48 hours,  
# NOTE- in the hns partition the default max run time is 7 days , so you wont need to include qos
 
#SBATCH --qos=bigmem

# We are submitting to the dev partition, there are several on sherlock: normal, gpu, owners, hns, bigmem (jobs requiring >64Gigs RAM) 
# 
#SBATCH -p bigmem
#################
#number of nodes you are requesting, the more you ask for the longer you wait

#SBATCH --nodes=1

#################
# --mem is memory per node; default is 4000 MB per CPU, remember to ask for enough mem to match your CPU request, since 
# sherlock automatically allocates 8 Gigs of RAM/CPU, if you ask for 8 CPUs you will need 32 Gigs of RAM, so either 
# leave --mem commented out or request >= to the RAM needed for your CPU request.

#SBATCH --mem=256GB

#################

# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
# Also, if you submit hundreds of jobs at once you will get hundreds of emails.
 
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
# Remember to change this to your email 
#SBATCH --mail-user=mincheol@stanford.edu


#now run normal batch commands
# note the "CMD BATCH is an R specific command IF YOU NEED MODULES
# module load R/3.3.0
# You can use srun if your job is parallel
#srun R CMD BATCH  ./rtest.R 

# otherwise: 
module load anaconda
source activate test_env
python isomap_clustering.py -n_neighbors 40 -n_components 40 -num_clusters 97
python isomap_clustering.py -n_neighbors 45 -n_components 40 -num_clusters 97
python isomap_clustering.py -n_neighbors 40 -n_components 45 -num_clusters 97
python isomap_clustering.py -n_neighbors 45 -n_components 50 -num_clusters 97 
python isomap_clustering.py -n_neighbors 45 -n_components 50 -num_clusters 97

python isomap_compute_distances.py -n_neighbors 40 -n_components 40 -num_clusters 97
python isomap_compute_distances.py -n_neighbors 45 -n_components 40 -num_clusters 97
python isomap_compute_distances.py -n_neighbors 40 -n_components 45 -num_clusters 97
python isomap_compute_distances.py -n_neighbors 45 -n_components 50 -num_clusters 97
python isomap_compute_distances.py -n_neighbors 45 -n_components 50 -num_clusters 97
source deactivate