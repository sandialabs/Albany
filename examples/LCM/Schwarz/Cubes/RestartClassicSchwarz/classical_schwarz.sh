
#!/bin/bash

#The following files must exist in this directory:

#cube0_in.exo       
#cube1_in.exo       
#input_schwarz_cube0_target.xml  
#input_schwarz_cube1_target.xml
#cube0_restart.xml  
#cube1_restart.xml  
#norm_displacements.m

if [ ! $1 ] ; then
    echo "This function requires 1 argument: # load steps (int)";
    exit
fi

echo "Initial cleanup..."
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
#initial cleanup: remove other old files
rm -rf displ0_old 
rm -rf displ1_old
rm -rf error  
echo "...cleanup done."

#tol_schwarz=0.000000000000001 #1e-15
tol_schwarz=0.001
echo "Schwarz convergence tolerance = $tol_schwarz"


#FIXME: set load correctly

#load step loop
for (( step=0; step<$1; step++ )); do 
   echo "Starting load step = $step..."
   echo "   Pre-processing..."
   #################  PRE-PROCESSING  ##############################
   #################  Cube 0  ######################################
   #Extract step^th snapshot from cube0_in.exo file
   ncks -d time_step,$step cube0_in.exo cube0_in_"$step".exo
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
   #create step^th input file for DTK run
   cp input_schwarz_cube0_target.xml input_schwarz_cube0_target_"$step".xml 
   #Replace name of exodus source file for step^th DTK input file
   exo_src_name=cube1_restart_out_"$step".exo 
   sed -i -e "s/cube1_restart_out.exo/$exo_src_name/g" input_schwarz_cube0_target_"$step".xml
   #Replace name of exodus target input file for step^th DTK input file
   exo_tgt_in_name=cube0_in_"$step".exo 
   sed -i -e "s/cube0_in.exo/$exo_tgt_in_name/g" input_schwarz_cube0_target_"$step".xml
   #Replace name of exodus target output file for step^th DTK input file
   exo_tgt_out_name=target_cube0_out_"$step".exo 
   sed -i -e "s/target_cube0_out.exo/$exo_tgt_out_name/g" input_schwarz_cube0_target_"$step".xml
   
   #################  Cube 1  ######################################
   #Extract step^th snapshot from cube1_in.exo file
   ncks -d time_step,$step cube1_in.exo cube1_in_"$step".exo
   #Create step^th input file for Albany run
   cp cube1_restart.xml cube1_restart_"$step".xml 
   #Replace name of exodus input file to restart from in step^th input file
   exo_in_name=cube1_in_"$step".exo
   sed -i -e "s/cube1_in.exo/$exo_in_name/g" cube1_restart_"$step".xml
   #Replace name of ouput file to restart from in step^th input file
   exo_out_name=cube1_restart_out_"$step".exo
   sed -i -e "s/cube1_restart_out.exo/$exo_out_name/g" cube1_restart_"$step".xml
   sed -i -e 's#<Parameter  name="Initial Value" type="double" value="0.0"/>#<Parameter  name="Initial Value" type="double" value="'$load_value'"/>#g' cube1_restart_"$step".xml
   sed -i -e 's#<Parameter  name="Min Value" type="double" value="0.0"/>#<Parameter  name="Min Value" type="double" value="'$load_value'"/>#g' cube1_restart_"$step".xml
   sed -i -e 's#<Parameter  name="Max Value" type="double" value="0.0"/>#<Parameter  name="Max Value" type="double" value="'$load_value'"/>#g' cube1_restart_"$step".xml
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
   echo "   ...pre-processing done."
   
   iterate_schwarz=1 #flag to tell code to continue Schwarz; if =0, Schwarz will stop
                     #this parameter should be reset to 1 in each load step 
   schwarz_iter=0 #Schwarz iteration number 
   while [ $iterate_schwarz -eq 1 ]; do 
     echo "   Running Albany on subdomain 0..."
     #################  ALBANY RUN FOR CUBE0  ##############################
     #run Albany with cube0_restart_"$step".xml input file
     ./AlbanyT cube0_restart_"$step".xml #this will write out an Exodus file with the name cube0_restart_out_$step.exo 
     echo "   ...Albany run on subdomain 0 done."

     echo "   Transferring solution in cube0 onto cube1 using DTK..."
     #################  DTK RUN FOR TRANSFERING CUBE0 SOLN TO CUBE1  ##############################
     #run DTK_Interp_Volume_to_NS executable with input_schwarz_cube1_target_"$step".xml
     ./DTK_Interp_Volume_to_NS --xml-in-file=input_schwarz_cube1_target_"$step".xml #this will write out an Exodus file with the name 
                                                                                  #target_cube0_out_$step.exo
     echo "   ...solution transfer from cube0 onto cube1 done."
  
     echo "   Post-processing of cube1 exo file..."
     #################  POST-PROCESSING AFTER DTK RUN ##############################
     #change time stamp in cube1_in_$step.exo to 0 to be consistent with DTK output target file which will have a time stamp of 0
     ncap2 -s 'time_whole=0*time_whole' cube1_in_"$step".exo tmp.exo
     mv tmp.exo cube1_in_"$step".exo 
     #merge target_cube1_out_$step.exo file with cube1_in_$step.exo
     #FIXME? do we want to merge cube1_in_"step".exo or cube1_out_$step".exo??
     ejoin target_cube1_out_"$step".exo cube1_in_"$step".exo #this produces a file ejoin-out.e
     mv ejoin-out.e cube1_restart_"$step".exo #copy ejoin-out.e to cube1_restart_$step.exo
     echo "   ...post-processing of cube1 exo file done."

     echo "   Running Albany on subdomain 1..."
     #################  ALBANY RUN FOR CUBE1  ##############################
     #run Albany with cube1_restart_"$step".xml input file
     ./AlbanyT cube1_restart_"$step".xml #this will write out an Exodus file with the name cube1_restart_out_$step.exo
     echo "   ...Albany run on subdomain 1 done."

     echo "   Transferring solution in cube1 onto cube0 using DTK..."
     #################  DTK RUN FOR TRANSFERING CUBE1 SOLN TO CUBE0  ##############################
     #run DTK_Interp_Volume_to_NS executable with input_schwarz_cube0_target_"$step".xml
     ./DTK_Interp_Volume_to_NS --xml-in-file=input_schwarz_cube0_target_"$step".xml #this will write out an Exodus file with the name 
                                                                                  #target_cube1_out_$step.exo
     echo "   ...solution transfer from cube1 onto cube0 done."

     echo "   Post-processing of cube0 exo file..."
     #################  POST-PROCESSING AFTER DTK RUN ##############################
     #change time stamp in cube0_in_$step.exo to 0 to be consistent with DTK output target file which will have a time stamp of 0
     ncap2 -s 'time_whole=0*time_whole' cube0_in_"$step".exo tmp.exo
     mv tmp.exo cube0_in_"$step".exo 
     #merge target_cube0_out_$step.exo file with cube0_in_$step.exo
     #FIXME? do we want to merge cube0_in_"step".exo or cube0_out_$step".exo??
     ejoin target_cube0_out_"$step".exo cube0_in_"$step".exo #this produces a file ejoin-out.e
     mv ejoin-out.e cube0_restart_"$step".exo #copy ejoin-out.e to cube1_restart_$step.exo
     echo "   ...post-processing of cube0 exo file done."

     echo "   Checking convergence..."
     #################  CHECK CONVERGENCE  #########################################
     matlab -nodesktop -nosplash -r "norm_displacements($schwarz_iter);quit;" #this will create an ascii file error containing the value of the error
     #read error from error file
     err=$(head -n 1 error)
     echo "  error = $err" 
     #check if error < tol_schwarz; if it is, the method is converged 
     if (($(echo $err '<=' $tol_schwarz | bc -l))); then 
       echo "  ...classical Schwarz converged after $schwarz_iter Schwarz iterations!"
       iterate_schwarz=0 
     else
       echo "   error = $err > tol_schwarz = $tol_schwarz"
       echo "  ...Schwarz failed to converge.  Continuing."
       #increment Schwarz iteration 
       let "schwarz_iter=schwarz_iter+1"
     fi
   done #while loop
   echo "...finished load step $step run!" 
done #load step loop 
