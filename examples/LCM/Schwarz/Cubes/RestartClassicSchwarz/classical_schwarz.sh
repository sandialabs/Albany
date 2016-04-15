
#!/bin/bash

if [ ! $1 ] ; then
    echo "This function requires 1 argument: # load steps (int)";
    exit
fi

for (( step=0; step<$1; step++ ))
do
   echo "Starting step $step of classical Schwarz with load = $step..."

   #################  ALBANY RUN FOR CUBE0  ##############################
   #Extract step^th snapshot from cube0_in.exo file
   ncks -d time_step,$step cube0_in.exo cube0_in_"$step".exo
   #Extract step^th snapshot from cube1_in.exo file
   ncks -d time_step,$step cube1_in.exo cube1_in_"$step".exo
   #Create step^th input file for Albany run
   cp cube0_restart.xml cube0_restart_"$step".xml 
   #Replace name of exodus input file to restart from in step^th input file
   exo_in_name=cube0_in_"$step".exo
   sed -i -e "s/cube0_in.exo/$exo_in_name/g" cube0_restart_"$step".xml
   #Replace name of ouput file to restart from in step^th input file
   exo_out_name=cube0_restart_out_"$step".exo
   sed -i -e "s/cube0_restart_out.exo/$exo_out_name/g" cube0_restart_"$step".xml
   #Set the load in the input file.  Note that load_value assumes the load is going from 0->1 right now.
   load_value=0.$step
   sed -i -e 's#<Parameter  name="Initial Value" type="double" value="0.0"/>#<Parameter  name="Initial Value" type="double" value="'$load_value'"/>#g' cube0_restart_"$step".xml
   sed -i -e 's#<Parameter  name="Min Value" type="double" value="0.0"/>#<Parameter  name="Min Value" type="double" value="'$load_value'"/>#g' cube0_restart_"$step".xml
   sed -i -e 's#<Parameter  name="Max Value" type="double" value="0.0"/>#<Parameter  name="Max Value" type="double" value="'$load_value'"/>#g' cube0_restart_"$step".xml
   #run Albany with cube0_restart_"$step".xml input file
   ./AlbanyT cube0_restart_"$step".xml

   #################  DTK RUN FOR TRANSFERING CUBE0 SOLN TO CUBE1  ##############################
   #create step^th input file for DTK run
   cp input_schwarz_cube1_target.xml input_schwarz_cube1_target_"$step".xml 
   #Replace name of exodus source file for step^th DTK input file
   exo_src_name=cube0_restart_out_"$step".exo 
   sed -i -e "s/cube0_restart_out.exo/$exo_src_name/g" input_schwarz_cube1_target_"$step".xml
   #Replace name of exodus target input file for step^th DTK input file
   exo_tgt_in_name=cube1_in_"$step".exo 
   sed -i -e "s/cube1_in.exo/$exo_tgt_in_name/g" input_schwarz_cube1_target_"$step".xml
   #Replace name of exodus target output file for step^th DTK input file
   exo_tgt_out_name=target_cube1_out_"$step".exo 
   sed -i -e "s/target_cube1_out.exo/$exo_tgt_out_name/g" input_schwarz_cube1_target_"$step".xml
   #run DTK_Interp_Volume_to_NS executable with input_schwarz_cube1_target_"$step".xml
   ./DTK_Interp_Volume_to_NS --xml-in-file=input_schwarz_cube1_target_"$step".xml

   echo "...done!"
done

