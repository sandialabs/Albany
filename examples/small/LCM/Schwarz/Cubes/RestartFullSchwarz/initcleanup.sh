
#!/bin/bash

#initial cleanup: remove all old *exo files
rm -rf target_cube0_out_*.exo
rm -rf target_cube1_out_*.exo
rm -rf cube0_restart_*.exo
rm -rf cube1_restart_*.exo
rm -rf cube0_restart_out_*.exo
rm -rf cube1_restart_out_*.exo
rm -rf cube0_in_*.exo
rm -rf cube1_in_*.exo
#initial cleanup: remove all old *xml files
rm -rf cube0_restart_*.xml
rm -rf cube1_restart_*.xml
rm -rf input_schwarz_cube*load*schwarz*.xml
#initial cleanup: remove other old files
rm -rf displ0_*
rm -rf displ1_*
rm -rf error*
rm -rf *.txt
rm -rf dtk_*

