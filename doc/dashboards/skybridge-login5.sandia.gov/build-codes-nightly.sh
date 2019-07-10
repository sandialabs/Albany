#!/bin/bash
## Do not put any commands or blank lines before the #SBATCH lines
export LCM_DIR=`pwd`
module use --append $LCM_DIR/Albany/doc/LCM/modulefiles
module load serial-intel-release
rm -rf trilinos-build-serial-intel-release
rm -rf trilinos-install-serial-intel-release
rm -rf albany-build-serial-intel-release
rm -rf Albany
rm -rf Trilinos
rm -rf *log 
rm -rf slurm*
./clean-config-build-dash.sh trilinos 8
./clean-config-build-dash.sh albany 8
mv albany-serial-intel-release.log albany-serial-intel-release-build.log 
