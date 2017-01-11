#!/bin/bash

module purge
module load gnu/4.9.2
module load openmpi-gnu/1.8
module load python/2.7

# Building Trilinos, add environmental variable and path
export REMOTE=/gscratch/jwfoulk/albany
export LD_LIBRARY_PATH=/opt/python-2.7/lib:$LD_LIBRARY_PATH

## Using Trilinos
LD_LIBRARY_PATH=$REMOTE/trilinos-install-gcc-release/lib:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=$REMOTE/lib:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=$REMOTE/lib64:$LD_LIBRARY_PATH
PATH=$REMOTE/trilinos-install-gcc-release/bin:$PATH

