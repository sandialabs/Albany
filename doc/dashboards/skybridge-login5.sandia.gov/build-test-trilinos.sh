#!/bin/bash
## Do not put any commands or blank lines before the #SBATCH lines
export TEST_DIR=`pwd`
rm -rf *log 
module use --append $TEST_DIR/Albany/doc/LandIce/modulefiles
module load serial-intel-release
rm -rf slurm*
./test-dash.sh trilinos
