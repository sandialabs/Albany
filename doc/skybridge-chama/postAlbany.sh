#!/bin/bash
module load lcm/sems
module load serial-gcc-release
module unload sems-python/2.7.9
module load canopy

# path to robustness suite
export ROBUSTNESS_SUITE=/ascldap/users/jwfoulk/SolidMechanics/SolidMechanicsExamples/LCM/CrystalPlasticity/robustnessTests

# must have includes for exodusII
PATH=$LCM_DIR/trilinos-install-serial-gcc-release/include:$PATH

## Python tools in robustness suite
export PYTHONPATH=$ROBUSTNESS_SUITE/Tools:$PYTHONPATH
export PATH=$ROBUSTNESS_SUITE/Tools:$PATH

## Account information needed for robustness suite
export MYWCID=fy150167
