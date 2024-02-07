#!/bin/bash
#SBATCH -N 1
##SBATCH -p c60-m0 
#SBATCH --time=8:00:00
#Note: This script assumes you have pre-loaded the required modules
#prior to entering the sbatch command. The env will propagate to the
#batch job.
#SBATCH --output=log-gcc.out  
if [ X"$SLURM_STEP_ID" = "X" -a X"$SLURM_PROCID" = "X"0 ]
then
  echo "print SLURM_JOB_ID = $SLURM_JOB_ID"
fi 

cd /home/projects/albany/nightlyCDashTrilinosBlake
source blake_gcc_modules.sh >& gcc_modules.out
bash nightly_cron_script_trilinos_blake_gcc_release.sh
bash nightly_cron_script_trilinos_blake_gcc_debug.sh
cd /home/projects/albany/nightlyCDashAlbanyBlake
bash nightly_cron_script_albany_blake_gcc_release.sh
bash nightly_cron_script_albany_blake_gcc_debug.sh
bash nightly_cron_script_albany_blake_gcc_sfad.sh sfad6
bash nightly_cron_script_albany_blake_gcc_sfad.sh sfad12
bash nightly_cron_script_albany_blake_gcc_sfad.sh sfad24
bash nightly_cron_script_mali_blake_gcc_release.sh

