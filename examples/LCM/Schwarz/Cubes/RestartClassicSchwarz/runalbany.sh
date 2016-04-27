
#!/bin/bash

if [ ! $3 ] ; then
    echo "This function requires 3 arguments: load step # (int), schwarz step # (int), cube # (int)";
    exit
fi

step=$1
schwarz_iter=$2
cube=$3

#run Albany with cube"$cube"_restart_load"$step"_schwarz"$schwarz_iter".xml input file
#the following will will write out an Exodus file with the name cube"$cube"_restart_out_load"$step"_schwarz"$schwarz"_iter.exo
./AlbanyT cube"$cube"_restart_load"$step"_schwarz"$schwarz_iter".xml >& albanyT_cube"$cube"_load"$step"_schwarz"$schwarz_iter"_out.txt
