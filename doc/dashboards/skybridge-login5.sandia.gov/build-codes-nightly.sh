#!/bin/bash
## Do not put any commands or blank lines before the #SBATCH lines
export LCM_DIR=`pwd`
rm -rf *log 
rm -rf Albany
rm -rf Trilinos
git clone git@github.com:trilinos/Trilinos.git >& trilinos-clone.log
cd Trilinos
git checkout develop
cd ..
git clone git@github.com:SNLComputation/Albany.git >& albany-clone.log 
module use --append $LCM_DIR/Albany/doc/LCM/modulefiles
module load serial-intel-release
rm -rf trilinos-build-serial-intel-release
rm -rf trilinos-install-serial-intel-release
rm -rf albany-build-serial-intel-release
rm -rf slurm*
./clean-update-config-build-dash.sh trilinos 8
./test-dash.sh trilinos
sbatch batch.openmpi.bash
