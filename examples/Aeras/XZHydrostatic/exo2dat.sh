#!/bin/bash
/scratch/jroverf/TrilinosSrc/Trilinos/build/install/bin/epu  -processor_count 30   -extension exo xzhydrostatic
/scratch/jroverf/TrilinosSrc/Trilinos/build/install/bin/exotxt xzhydrostatic.exo xzhydrostatic.txt
./txt2dat.py > xzhydrostatic.dat 
# scp xzhydrostatic.dat face:tec

