#!/bin/bash
/scratch/jroverf/TrilinosSrc/install/bin/epu  -processor_count 20   -extension exo xzhydrostatic
/scratch/jroverf/TrilinosSrc/install/bin/exotxt xzhydrostatic.exo xzhydrostatic.txt
./txt2dat.py > xzhydrostatic.dat 
scp xzhydrostatic.dat face:tec

