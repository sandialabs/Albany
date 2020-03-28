#!/bin/bash
## Do not put any commands or blank lines before the #SBATCH lines
#SBATCH --nodes=1                    # Number of nodes - all cores per node are allocated to the job
#SBATCH --time=01:15:00               # Wall clock time (HH:MM:SS) - once the job exceeds this time, the job will be terminated (default is 5 minutes)
#SBATCH --account=FY180068              # WC ID
#SBATCH --job-name=AlbNightly               # Name of job
#SBATCH --partition=short             # partition/queue name: short or batch
                                      #            short: 4hrs wallclock limit
                                      #            batch: nodes reserved for > 4hrs (default)
                                      #            short,batch: job will move to batch if short is full

#SBATCH --qos=normal                  # Quality of Service: long, large, priority or normal
                                      #           normal: request up to 48hrs wallclock (default)
                                      #           long:   request up to 96hrs wallclock and no larger than 64nodes
                                      #           large:  greater than 50% of cluster (special request)
                                      #           priority: High priority jobs (special request)

##SBATCH --mail-type=ALL
##SBATCH --mail-user=ikalash@sandia.gov
#nodes=$SLURM_JOB_NUM_NODES           # Number of nodes - the number of nodes you have requested (for a list of SLURM environment variables see "man sbatch")
#cores=1                             # Number MPI processes to run on each node (a.k.a. PPN)
                                     # tlcc2 has 16 cores per node
# using openmpi-intel/1.10
cd /home/ikalash/LCM
export LCM_DIR=`pwd`
module use --append $LCM_DIR/Albany/doc/LCM/modulefiles
module load serial-intel-release
./clean-update-config-build-dash.sh albany 8
mv albany-serial-intel-release.log albany-serial-intel-release-build.log 
./test-dash.sh albany 
bash process_results.sh
bash send_email.sh
# Note 1: This will start ($nodes * $cores) total MPI processes using $cores per node.
#           If you want some other number of processes, add "-np N" after the mpiexec, where N is the total you want.
#           Example:  mpiexec -np 24  ......(for a 2 node job, this will load 16 processes on the first node and 8 processes on the second node)
#           If you want a specific number of process to run on each node, (thus increasing the effective memory per core), use the --npernode option.
#           Example: mpiexec -np 24 --npernode 12  ......(for a 2 node job, this will load 12 processes on each node)

# The default version of Open MPI is version 1.10.

# For openmpi 1.10: mpiexec --bind-to core --npernode 8 --n PUT_THE_TOTAL_NUMBER_OF_MPI_PROCESSES_HERE /path/to/executable [--args...]

# To submit your job, do:
# sbatch <script_name>
#
#The slurm output file will by default, be written into the directory you submitted your job from  (slurm-JOBID.out)
