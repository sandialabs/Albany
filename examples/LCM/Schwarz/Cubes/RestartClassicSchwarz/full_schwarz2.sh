
#!/bin/bash

#The following files must exist in this directory:

#cube0_in.exo       
#cube1_in.exo       
#input_schwarz_cube0_target.xml  
#input_schwarz_cube1_target.xml
#cube0_restart.xml  
#cube1_restart.xml  
#norm_displacements.m

if [ ! $2 ] ; then
    echo "This function requires 2 arguments: # load steps (int), Schwarz convergence tolerance (double)";
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
rm -rf displ0_*
rm -rf displ1_*
rm -rf error*
rm -rf *.txt
rm -rf dtk_*  
echo "...cleanup done."

tol_schwarz=$2
#convert from scientific "e" notation to notation readable by the bc bash tool
tol_schwarz=`echo ${tol_schwarz} | sed -e 's/[eE]+*/\\*10\\^/'`
echo "Schwarz convergence tolerance = $tol_schwarz"
#TODO? Set alternative convergence tolerance, e.g., max # Schwarz iterations
#FIXME: save more history for debugging (e.g., history of displ*_old's, not just the past one).
#NOTE: the code assumes cube0_in.exo and cube1_in.exo have $1 snapshots.  I think this is necessary 
#if there is a time-dependent boundary condition; otherwise it is not and the code below can be rewritten 
#easily to handle this case. 
#TODO: can Alejandro or Coleman create input  *exo file for this problem that contains only dirichlet_field field
#to make sure things still work with a restart from a trivial initial condition? 


#load step loop
for (( step=0; step<$1; step++ )); do
 
   echo "Starting load step = $step..."
   iterate_schwarz=1 #flag to tell code to continue Schwarz; if iterate_schwarz==0, Schwarz will stop
                     #this parameter should be reset to 1 in each load step 
   schwarz_iter=0 #Schwarz iteration number 

   #Set the load value.  Note that load_value assumes the load is going from 0->1 right now.
   load_value=0.$step

   #integer for keeping track of how many schwarz iterations were in the previous schwarz step
   num_schwarz_iter_prev=0 

   while [ $iterate_schwarz -eq 1 ]; do
     
     echo "   Starting schwarz iter = $schwarz_iter..."
     
     #initial load step: extract first snapshot from cube0_in.exo and cube1_in.exo
     if [ $step -eq 0 ]; then
       ncks -d time_step,$step cube0_in.exo cube0_in_load"$step"_schwarz"$schwarz_iter".exo
       ncks -d time_step,$step cube1_in.exo cube1_in_load"$step"_schwarz"$schwarz_iter".exo
     fi
     #FIXME: create cube*_in* files for step > 0.  For these, the Dirichlet data should be from step $step
     #in cube0_in.exo and cube1_in.exo; but the displacement should be from 
     #  - cube0_restart_load$step_schwarz($schwarz_iter-1).exo (similarly for cube1) if load step has not been incremented
     #  - cube0_restart_load$(step-1)_schwarz(last schwarz iter #).exo.
    
     #################  PRE-PROCESSING  ##############################
     echo "      Starting pre-processing..."
     bash preprocess.sh $step $schwarz_iter $load_value 0 #cube0 
     bash preprocess.sh $step $schwarz_iter $load_value 1 #cube1
     echo "      ...pre-processing done." 
     ##################################################################

     #################  ALBANY RUN FOR CUBE0  #########################
     echo "      Running Albany on cube0..."
     bash runalbany.sh $step $schwarz_iter 0 
     echo "      ...Albany cube0 run done."
     ##################################################################
     
     #################  DTK TRANSFER FROM CUBE0 TO CUBE1  #############
     echo "      Transferring solution in cube0 onto cube1 using DTK..."
     bash dtktransfer.sh $step $schwarz_iter 1 
     echo "      ...DTK transfer from cube0 onto cube1 done."
     ##################################################################

     #################  POST-DTK RUN PROCESSING FOR CUBE1  ############
     echo "      Starting post-DTK run for cube1 processing..."
     bash postdtkprocess.sh $step $schwarz_iter 1
     echo "      ...post-DTK cube1 run done."
     ##################################################################

     #################  ALBANY RUN FOR CUBE1  #########################
     echo "      Running Albany on cube1..."
     bash runalbany.sh $step $schwarz_iter 1
     echo "      ...Albany cube1 run done."
     ##################################################################

     #################  DTK TRANSFER FROM CUBE1 TO CUBE0  #############
     echo "      Transferring solution in cube1 onto cube0 using DTK..."
     bash dtktransfer.sh $step $schwarz_iter 0 
     echo "      ...DTK transfer from cube1 onto cube0 done."
     ##################################################################

     #################  POST-DTK RUN PROCESSING FOR CUBE0  ############
     echo "      Starting post-DTK run for cube0 processing..."
     bash postdtkprocess.sh $step $schwarz_iter 0
     echo "      ...post-DTK cube0 run done."
     ##################################################################

     #################  CHECK CONVERGENCE OF SCHWARZ  #################
     echo "      Checking if Schwarz converged..."
     #the following will create an ascii file error containing the value of the error as well as ascii files with the displacements
     matlab -nodesktop -nosplash -r "norm_displacements2($schwarz_iter, $step, $num_schwarz_iter_prev);quit;"
     #read error from error file
     err=$(head -n 1 error)
     #convert from scientific "e" notation to notation readable by the bc bash tool
     err=`echo ${err} | sed -e 's/[eE]+*/\\*10\\^/'`
     echo "      error = $err" 
     #check if error < tol_schwarz; if it is, the method is converged 
     if (($(echo $err '<=' $tol_schwarz | bc -l))); then
       num_schwarz_iter_prev=$schwarz_iter
       echo "     ...classical Schwarz converged after $num_schwarz_iter_prev Schwarz iterations!"
       iterate_schwarz=0
     else
       echo "     error = $err > tol_schwarz = $tol_schwarz"
       echo "     ...Schwarz failed to converge.  Continuing."
       #increment Schwarz iteration 
       let "schwarz_iter=schwarz_iter+1"
     fi

     ##################################################################

   done #while loop
   echo "...finished load step $step run!" 
done #load step loop 
