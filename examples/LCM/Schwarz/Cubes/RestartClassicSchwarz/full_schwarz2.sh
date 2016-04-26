
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
rm -rf displ0_old 
rm -rf displ1_old
rm -rf displ0_current*
rm -rf displ1_current*
rm -rf error*
rm -rf *.txt 
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

   while [ $iterate_schwarz -eq 1 ]; do
     
     echo "   Starting schwarz iter = $schwarz_iter..."
     
     #initial load step: extract first snapshot from cube0_in.exo and cube1_in.exo
     if [ $step -eq 0 ]; then
       ncks -d time_step,$step cube0_in.exo cube0_in_load"$step"_schwarz"$schwarz_iter".exo
       ncks -d time_step,$step cube1_in.exo cube1_in_load"$step"_schwarz"$schwarz_iter".exo
     fi
    
     #################  PRE-PROCESSING  ##############################
     echo "      Starting pre-processing..."
     bash preprocess.sh $step $schwarz_iter $load_value 0 #cube0 
     bash preprocess.sh $step $schwarz_iter $load_value 1 #cube1
     echo "      pre-processing done." 
     ##################################################################

     #################  ALBANY RUN FOR CUBE0  #########################
     echo "      Running Albany on subdomain 0..."
     bash runalbany.sh $step $schwarz_iter 0 
     echo "      Albany subdomain 0 run done..."
     ##################################################################

     #FIXME: the following line will need to change 
     let "iterate_schwarz=0"
     echo "   ...finished schwarz iter = $schwarz_iter" 
    
   done #while loop
   echo "...finished load step $step run!" 
done #load step loop 
