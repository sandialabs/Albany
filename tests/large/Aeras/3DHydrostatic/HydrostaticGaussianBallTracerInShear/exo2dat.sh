#!/bin/bash
/scratch/jroverf/TrilinosSrc/Trilinos/build/install/bin/epu -steps 1:1000000:10   -processor_count 30   -extension exo sphere10_quad9
/scratch/jroverf/TrilinosSrc/Trilinos/build/install/bin/exotxt sphere10_quad9.exo sphere10_quad9.txt
./txt2dat.py > sphere10_quad9.dat 
scp sphere10_quad9.dat face:tec

