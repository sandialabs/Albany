
#!/bin/bash

#The following files must exist in this directory:

#cube0_in.exo       
#cube1_in.exo       
#input_schwarz_cube0_target.xml  
#input_schwarz_cube1_target.xml
#cube0_restart.xml  
#cube1_restart.xml  
#norm_displacements.m

if [ ! $3 ] ; then
    echo "This function requires 3 arguments: # load steps (int), Schwarz convergence tolerance (double), max number Schwarz iters (int)";
    exit
fi

STARTTIME=$(date +%s)

echo "Initial cleanup..."
bash initcleanup.sh
echo "...cleanup done."

tol_schwarz=$2
#convert from scientific "e" notation to notation readable by the bc bash tool
tol_schwarz=`echo ${tol_schwarz} | sed -e 's/[eE]+*/\\*10\\^/'`
echo "Schwarz convergence tolerance = $tol_schwarz"
max_schwarz_iter=$3
let "max_schwarz_iter=max_schwarz_iter-1"

#integer for keeping track of how many schwarz iterations were in the previous schwarz step
num_schwarz_iter_prev=0 


#load step loop
for (( step=0; step<$1; step++ )); do
 
   echo "Starting load step = $step..."

   echo "   num_schwarz_iter_prev = $num_schwarz_iter_prev"

   iterate_schwarz=1 #flag to tell code to continue Schwarz; if iterate_schwarz==0, Schwarz will stop
                     #this parameter should be reset to 1 in each load step 
   schwarz_iter=0 #Schwarz iteration number 

   #Set the load value.  Note that load_value assumes the load is going from 0->1 right now.
   load_value=0.$step

   while [ $iterate_schwarz -eq 1 ]; do
     
     echo "   Starting schwarz iter = $schwarz_iter..."
     
     #################  PRE-PROCESSING  ##############################
     echo "      Starting pre-processing..."
     #the following creates input files:
     #- cube0_restart_load"$step"_schwarz"$schwarz_iter".xml
     #- cube1_restart_load"$step"_schwarz"$schwarz_iter".xml
     #- target_cube0_out_load"$step"_schwarz"$schwarz_iter".exo
     #- target_cube1_out_load"$step"_schwarz"$schwarz_iter".exo
     bash preprocess.sh $step $schwarz_iter $load_value 0 #cube0 
     bash preprocess.sh $step $schwarz_iter $load_value 1 #cube1
     echo "      ...pre-processing done." 
     ##################################################################
     
     #extract first snapshot from cube0_in.exo and cube1_in.exo.
     #the latter is required for the first DTK transfer step.
     ncks -d time_step,$step cube0_in.exo cube0_in_load"$step"_schwarz"$schwarz_iter".exo
     ncks -d time_step,$step cube1_in.exo cube1_in_load"$step"_schwarz"$schwarz_iter".exo

     #################  DTK TRANSFER FROM CUBE1 TO CUBE0  #############
     #DTK transfer does not make sense for 1st load step
     #NOTE: the logic here implicitly assumes the 1st load step is trivial (solutions are all 0). 
     if [ $step -gt 0 ]; then 
       echo "      Transferring solution in cube1 onto cube0 using DTK..."
       if [ $schwarz_iter -eq 0 ]; then
         let "prev_step=step-1"
         cp cube1_restart_out_load"$prev_step"_schwarz"$num_schwarz_iter_prev".exo cube1_restart_out_load"$step"_schwarz"$schwarz_iter".exo
       else  
         let "prev_schwarz_iter=schwarz_iter-1"
         cp cube1_restart_out_load"$step"_schwarz"$prev_schwarz_iter".exo cube1_restart_out_load"$step"_schwarz"$schwarz_iter".exo
       fi
       #we run DTK_Interp_Volume_to_NS with input file input_schwarz_cube1_target_load"$step"_schwarz"$schwarz_iter".xml
       #the output from the run is redirected to dtk_cube0_load"$step"_schwarz"$schwarz_iter"_out.txt
       #the input source mesh is cube1_restart_out_load"$step"_schwarz"$schwarz_iter".exo
       #the input target mesh is cube0_in_load"$step"_schwarz"$schwarz_iter".exo
       #the output target mesh is target_cube0_out_load"$step"_schwarz"$schwarz_iter".exo
       #interpolation is performed for the disp field from source input mesh onto the dirichlet_field field 
       #in the target input mesh and written to the dirichlet_field field of target output mesh.
       bash dtktransfer.sh $step $schwarz_iter 0 
       echo "      ...DTK transfer from cube1 onto cube0 done."
       ##################################################################

       #################  POST-DTK RUN PROCESSING FOR CUBE0  ############
       echo "      Starting post-DTK run for cube0 processing..."
       source_file=target_cube0_out_load"$step"_schwarz"$schwarz_iter".exo
       target_file=cube0_in_load"$step"_schwarz"$schwarz_iter".exo
       #here, we replace the dirichlet_field in cube0_in_load"$step"_schwarz"$schwarz_iter".exo 
       #with the dirichlet_field in target_cube0_out_load"$step"_schwarz"$schwarz_iter".exo
       python replace_nodal_field.py --target_file $target_file --target_index 1 --source_file $source_file --source_index 1
       python replace_nodal_field.py --target_file $target_file --target_index 2 --source_file $source_file --source_index 2
       python replace_nodal_field.py --target_file $target_file --target_index 3 --source_file $source_file --source_index 3
       #here, we replace the disp field in cube0_in_load"$step"_schwarz"$schwarz_iter".exo
       #with the dirichlet_field in target_cube0_out_load"$step"_schwarz"$schwarz_iter".exo for restarts
       python replace_nodal_field.py --target_file $target_file --target_index 4 --source_file $source_file --source_index 1
       python replace_nodal_field.py --target_file $target_file --target_index 5 --source_file $source_file --source_index 2
       python replace_nodal_field.py --target_file $target_file --target_index 6 --source_file $source_file --source_index 3
       echo "      ...post-DTK cube0 run done."
       ##################################################################
     fi
    
     #################  ALBANY RUN FOR CUBE0  #########################
     echo "      Running Albany on cube0..."
     #we run Albany with input file cube0_restart_load"$step"_schwarz"$schwarz_iter".xml
     #the output from the run is redirected to albanyT_cube0_load"$step"_schwarz"$schwarz_iter"_out.txt
     #the input Exodus mesh is cube0_in_load"$step"_schwarz"$schwarz_iter".exo
     #the output Exodus mesh is cube0_restart_out_load"$step"_schwarz"$schwarz_iter".exo.  it has 1 snapshot.
     bash runalbany.sh $step $schwarz_iter 0 
     echo "      ...Albany cube0 run done."
     ##################################################################
     
     #################  DTK TRANSFER FROM CUBE0 TO CUBE1  #############
     echo "      Transferring solution in cube0 onto cube1 using DTK..."
     #we run DTK_Interp_Volume_to_NS with input file input_schwarz_cube0_target_load"$step"_schwarz"$schwarz_iter".xml
     #the output from the run is redirected to dtk_cube1_load"$step"_schwarz"$schwarz_iter"_out.tx
     #the input source mesh is cube0_restart_out_load"$step"_schwarz"$schwarz_iter".exo
     #the input target mesh is cube1_in_load"$step"_schwarz"$schwarz_iter".exo
     #the output target mesh is target_cube1_out_load"$step"_schwarz"$schwarz_iter".exo
     #interpolation is performed for the disp field from source input mesh onto the dirichlet_field field 
     #in the target input mesh and written to the dirichlet_field field of target output mesh.
     bash dtktransfer.sh $step $schwarz_iter 1 
     echo "      ...DTK transfer from cube0 onto cube1 done."
     ##################################################################

     #################  POST-DTK RUN PROCESSING FOR CUBE1  ############
     echo "      Starting post-DTK run for cube1 processing..."
     source_file=target_cube1_out_load"$step"_schwarz"$schwarz_iter".exo
     target_file=cube1_in_load"$step"_schwarz"$schwarz_iter".exo
     #here, we replace the dirichlet_field in cube1_in_load"$step"_schwarz"$schwarz_iter".exo 
     #with the dirichlet_field in target_cube1_out_load"$step"_schwarz"$schwarz_iter".exo
     python replace_nodal_field.py --target_file $target_file --target_index 1 --source_file $source_file --source_index 1
     python replace_nodal_field.py --target_file $target_file --target_index 2 --source_file $source_file --source_index 2
     python replace_nodal_field.py --target_file $target_file --target_index 3 --source_file $source_file --source_index 3
     #here, we replace the disp field in cube1_in_load"$step"_schwarz"$schwarz_iter".exo
     #with the dirichlet_field in target_cube1_out_load"$step"_schwarz"$schwarz_iter".exo for restarts
     python replace_nodal_field.py --target_file $target_file --target_index 4 --source_file $source_file --source_index 1
     python replace_nodal_field.py --target_file $target_file --target_index 5 --source_file $source_file --source_index 2
     python replace_nodal_field.py --target_file $target_file --target_index 6 --source_file $source_file --source_index 3
     echo "      ...post-DTK cube1 run done."
     ##################################################################

     #################  ALBANY RUN FOR CUBE1  #########################
     echo "      Running Albany on cube1..."
     #we run Albany with input file cube1_restart_load"$step"_schwarz"$schwarz_iter".xml
     #the output from the run is redirected to albanyT_cube1_load"$step"_schwarz"$schwarz_iter"_out.txt
     #the input Exodus mesh is cube1_in_load"$step"_schwarz"$schwarz_iter".exo
     #the output Exodus mesh is cube1_restart_out_load"$step"_schwarz"$schwarz_iter".exo.  it has 1 snapshot.
     bash runalbany.sh $step $schwarz_iter 1
     echo "      ...Albany cube1 run done."
     ##################################################################

     #################  CHECK CONVERGENCE OF SCHWARZ  #################
     echo "      Checking if Schwarz converged..."
     #the following will create an ascii file containing the value of the error as well as ascii files with the displacements
     python norm_displacements.py --schwarz_no $schwarz_iter --step_no $step --num_schwarz_prev $num_schwarz_iter_prev
     #read error from ascii file
     err=$(head -n 1 error)
     #convert from scientific "e" notation to notation readable by the bc bash tool
     err=`echo ${err} | sed -e 's/[eE]+*/\\*10\\^/'`
     echo "      error = $err" 
     #check if error < tol_schwarz; if it is, the method is converged 
     if (($(echo $err '<=' $tol_schwarz | bc -l))); then
       num_schwarz_iter_prev=$schwarz_iter
       echo "     ...full Schwarz converged after $num_schwarz_iter_prev Schwarz iterations!"
       iterate_schwarz=0
     else
       echo "     error = $err > tol_schwarz = $tol_schwarz"
       if [ $schwarz_iter -lt $max_schwarz_iter ]; then
         echo "     ...Schwarz failed to converge.  Continuing."
         #increment Schwarz iteration 
         let "schwarz_iter=schwarz_iter+1"
       else 
         echo "     ...Schwarz failed to converge, but max number of Schwarz iterations (= $max_schwarz_iter) hit.  Proceeding to next load step."
         num_schwarz_iter_prev=$schwarz_iter
         iterate_schwarz=0
       fi
     fi
     #if [ $schwarz_iter -eq 2 ]; then
     #   exit
     #fi
   
     ##################################################################

   done #while loop
   echo "...finished load step $step run!" 
done #load step loop 

ENDTIME=$(date +%s)
echo "Full Schwarz took $(($ENDTIME - $STARTTIME)) seconds to complete."

echo "Beginning error processing..." 
bash process_errors.sh $1
echo "...error processing done." 
echo "Beginning error plotting..."
python plot_errs.py --num_load_steps $1  
echo "...error plotting done."
