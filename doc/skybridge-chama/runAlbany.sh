#!/bin/bash

module purge
module load gnu/4.9.2
module load openmpi-gnu/1.8
module load python/2.7

# Declare path to repository (REMOTE), dated build (REMOTE_EXEC), 
# and robustness suite (in SolidMechanicsExamples)
export REMOTE=/gscratch/jwfoulk/albany
export REMOTE_EXEC=/gscratch/jwfoulk/albany
export ROBUSTNESS_SUITE=/ascldap/users/jwfoulk/SolidMechanicsExamples/LCM/CrystalPlasticity/robustnessTests

# needed path for python
export LD_LIBRARY_PATH=/opt/python-2.7/lib:$LD_LIBRARY_PATH

## declare library paths from dated build
LD_LIBRARY_PATH=$REMOTE_EXEC/trilinos-install-gcc-release/lib:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=$REMOTE_EXEC/lib:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=$REMOTE_EXEC/lib64:$LD_LIBRARY_PATH
PATH=$REMOTE_EXEC/trilinos-install-gcc-release/bin:$PATH

## Python paths for exodus.py within dated build
export PYTHONPATH=$REMOTE_EXEC/trilinos-install-gcc-release/lib:$PYTHONPATH

## Python utils from LCM (requires repository declared in REMOTE)
export LCM_DIR=$REMOTE/src
export PYTHONPATH=$LCM_DIR/Albany/src/LCM/utils/python:$PYTHONPATH
export PATH=$LCM_DIR/Albany/src/LCM/utils/python:$PATH
export PYTHONPATH=$LCM_DIR/Albany/src/LCM/utils/python/lcm_postprocess:$PYTHONPATH
export PATH=$LCM_DIR/Albany/src/LCM/utils/python/lcm_postprocess:$PATH

## Python tools in robustness suite
export PYTHONPATH=$ROBUSTNESS_SUITE/Tools:$PYTHONPATH
export PATH=$ROBUSTNESS_SUITE/Tools:$PATH

## Account information needed for robustness suite
export MYWCID=fy150167
