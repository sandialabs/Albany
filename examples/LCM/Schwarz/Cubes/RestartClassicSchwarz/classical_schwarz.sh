
#!/bin/bash

if [ ! $1 ] ; then
    echo "This function requires 1 argument: # load steps (int)";
    exit
fi

for (( step=0; step<$1; step++ ))
do
   echo "Starting step $step of classical Schwarz with load = $step..."
   #Extract step^th snapshot from cube0_in.exo file
   ncks -d time_step,$step cube0_in.exo cube0_in_"$step".exo
   #Create step^th input file
   cp cube0_restart.xml cube0_restart_"$step".xml 
   #Replace name of input file to restart from in step^th input file
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
   echo "...done!"
done

