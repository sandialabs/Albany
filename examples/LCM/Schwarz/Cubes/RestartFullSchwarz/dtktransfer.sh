
#!/bin/bash

if [ ! $3 ] ; then
    echo "This function requires 3 arguments: load step # (int), schwarz step # (int), cube # (int)";
    exit
fi

step=$1
schwarz_iter=$2
cube=$3

#run DTK_Interp_Volume_to_NS executable with input_schwarz_cube"$cube"_target_"$step".xml
#the folowing will write out an Exodus file with the name target_cube"$cube"_out_load"$step"_schwarz"$schwarz_iter".exo
./DTK_Interp_Volume_to_NS --xml-in-file=input_schwarz_cube"$cube"_target_load"$step"_schwarz"$schwarz_iter".xml >& dtk_cube"$cube"_load"$step"_schwarz"$schwarz_iter"_out.txt 

