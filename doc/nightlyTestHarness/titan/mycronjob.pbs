#!/bin/bash -l
#PBS -A CLI062
#PBS -l walltime=01:00:00,nodes=1,partition=login 

module load git cmake

cd piscees/AlbanyContinuousIntegration/scripts

# and one script to run them all
./run_master.sh set_titan_env.in 

# recursively resubmit this script
X=$( date -d "3 days 3 am" "+%Y%m%d%H%M" )
qsub -a $X ~/mycronjob.pbs
