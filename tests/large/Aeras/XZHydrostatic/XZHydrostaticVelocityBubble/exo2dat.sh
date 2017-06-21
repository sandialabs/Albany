#!/bin/bash
/scratch/jroverf/TrilinosSrc/Trilinos/build/install/bin/epu -steps 1:1000000:10  -processor_count 30   -extension exo xzhydrostatic
/scratch/jroverf/TrilinosSrc/Trilinos/build/install/bin/exotxt xzhydrostatic.exo xzhydrostatic.txt
./txt2dat.py > xzhydrostatic.dat 
scp xzhydrostatic.dat face:tec

