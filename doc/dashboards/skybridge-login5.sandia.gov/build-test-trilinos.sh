#!/bin/bash
## Do not put any commands or blank lines before the #SBATCH lines
export LCM_DIR=`pwd`
rm -rf *log 
module use --append $LCM_DIR/Albany/doc/LCM/modulefiles
module load serial-intel-release
rm -rf slurm*
./test-dash.sh trilinos
