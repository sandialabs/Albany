
#!/bin/bash

#The following files must exist in this directory:

#cube0_in.exo       
#cube1_in.exo       
#input_schwarz_cube0_target.xml  
#input_schwarz_cube1_target.xml
#cube0_restart.xml  
#cube1_restart.xml  

if [ ! $4 ] ; then
    echo "This function requires 4 arguments: load step # (int), schwarz step # (int), load value (double), cube # (int)";
    exit
fi

step=$1
schwarz_iter=$2
load_value=$3
cube=$4

echo "         load step = " $step 
echo "         schwarz iter = " $schwarz_iter
echo "         load_value = " $load_value
echo "         cube = " $cube

if [ $cube -eq 0 ]; then
  cube_coupled=1
elif [ $cube -eq 1 ]; then
  cube_coupled=0
else 
  echo "Value of cube = " $cube " is undefined!  Valid values are 0 and 1!"
  exit
fi 

echo "         cube_coupled = " $cube_coupled


#################  PRE-PROCESSING  ##############################
#Create input file for Albany run for load step $step and schwarz step $schwarz_iter
init_xml_name=cube"$cube"_restart.xml
cp $init_xml_name cube"$cube"_restart_load"$step"_schwarz"$schwarz_iter".xml
#Replace name of exodus input file to restart from in step^th input file
exo_in_name=cube"$cube"_in_load"$step"_schwarz"$schwarz_iter".exo
init_exo_in_name=cube"$cube"_in.exo
sed -i -e "s/$init_exo_in_name/$exo_in_name/g" cube"$cube"_restart_load"$step"_schwarz"$schwarz_iter".xml
#Replace name of output file to restart from for load step $step and schwarz step $schwarz_iter
exo_out_name=cube"$cube"_restart_out_load"$step"_schwarz"$schwarz_iter".exo
init_exo_out_name=cube"$cube"_restart_out_exo
sed -i -e "s/$init_exo_out_name/$exo_out_name/g" cube"$cube"_restart_load"$step"_schwarz"$schwarz_iter".xml
#Set the load in the input file.  
sed -i -e 's#<Parameter  name="Initial Value" type="double" value="0.0"/>#<Parameter  name="Initial Value" type="double" value="'$load_value'"/>#g' cube"$cube"_restart_load"$step"_schwarz"$schwarz_iter".xml
sed -i -e 's#<Parameter  name="Min Value" type="double" value="0.0"/>#<Parameter  name="Min Value" type="double" value="'$load_value'"/>#g' cube"$cube"_restart_load"$step"_schwarz"$schwarz_iter".xml
sed -i -e 's#<Parameter  name="Max Value" type="double" value="0.0"/>#<Parameter  name="Max Value" type="double" value="'$load_value'"/>#g' cube"$cube"_restart_load"$step"_schwarz"$schwarz_iter".xml
#create input file for DTK run in load step $step and schwarz step $schwarz_iter
cp input_schwarz_cube"$cube"_target.xml input_schwarz_cube"$cube"_target_load"$step"_schwarz"$schwarz_iter".xml
#Replace name of exodus source file for DTK input file in load step $step and schwarz step $schwarz_iter
exo_src_name=cube"$cube_coupled"_restart_out_load"$step"_schwarz"$schwarz_iter".exo
init_exo_src_name=cube"$cube_coupled"_restart_out.exo
sed -i -e "s/$init_exo_src_name/$exo_src_name/g" input_schwarz_cube"$cube"_target_load"$step"_schwarz"$schwarz_iter".xml
#Replace name of exodus target input file for DTK input file in load step $step and schwarz step $schwarz_iter
exo_tgt_in_name=cube"$cube"_in_load"$step"_schwarz"$schwarz_iter".exo
sed -i -e "s/$init_exo_in_name/$exo_tgt_in_name/g" input_schwarz_cube"$cube"_target_load"$step"_schwarz"$schwarz_iter".xml
#Replace name of exodus target output file for DTK input file in load step $step and schwarz_step $schwarz_iter
exo_tgt_out_name=target_cube"$cube"_out_load"$step"_schwarz"$schwarz_iter".exo
init_exo_tgt_out_name=target_cube"$cube"_out.exo
sed -i -e "s/$init_exo_tgt_out_name/$exo_tgt_out_name/g" input_schwarz_cube"$cube"_target_load"$step"_schwarz"$schwarz_iter".xml

