#!/bin/bash
## Do not put any commands or blank lines before the #SBATCH lines
#SBATCH --nodes=1                    # Number of nodes - all cores per node are allocated to the job
##SBATCH --ntasks-per-node=36          # Number of cores per node
#SBATCH --time=01:00:00               # Wall clock time (HH:MM:SS) - once the job exceeds this time, the job will be terminated (default is 5 minutes)
#SBATCH --account=FY230074  # WC ID
#SBATCH --job-name=AlbNightlyRun               # Name of job
#SBATCH --partition=short       # partition/queue name: short or batch
                                      #            short: 4hrs wallclock limit
                                      #            batch: nodes reserved for > 4hrs (default)
                                      #            short,batch: job will move to batch if short is full

#SBATCH --qos=normal                  # Quality of Service: long, large, priority or normal
                                      #           normal: request up to 48hrs wallclock (default)
                                      #           long:   request up to 96hrs wallclock and no larger than 64nodes
                                      #           large:  greater than 50% of cluster (special request)
                                      #           priority: High priority jobs (special request)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ikalash@sandia.gov

nodes=$SLURM_JOB_NUM_NODES           # Number of nodes - the number of nodes you have requested (for a list of SLURM environment variables see "man sbatch")
cores=36                             # Number MPI processes to run on each node (a.k.a. PPN)
                                     # cts1 has 36 cores per node
bash nightly_cron_script_albany_attaway_intel_serial_run.sh  

