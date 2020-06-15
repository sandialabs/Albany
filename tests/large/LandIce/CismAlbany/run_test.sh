
#!/bin/bash

#IKT, WARNING: the following 2 lines are specific to Irina Tezaur's machine, camobap!
#They need to be changed for other machines! 
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib64/openmpi/lib:/usr/lib
export PATH=$PATH:/install/ParaView-5.8.0-Linux-Sandia/bin:/install/cubit-15.4b:/usr/local/MATLAB/R2019b/bin:/install/seacas-utils:/usr/lib64/openmpi/bin

rm -rf *exo*
rm -rf albanyMesh/*exo*

# CISM-ALBANY

# run cism-albany after modifying (if needed) the paths of the input nc "name" file and the "dycore_input_file" in the file inputFiles/cism-albanyT.config.
cd inputFiles
rm -rf *exo* 
mpiexec -np 8 ../cism_driver/cism_driver cism-albanyT.config
epu --auto greenland_cism-albanyT.exo.8.0
cd ..

# [optional] if you run the above on multiple processors, you need to merge the exodus files into one:
#epu --auto greenland_cism-albanyT.exo.4.

#note that if you diff the original greenland.nc file and the one stored by cism greenland_cism-albanyT.nc, beta changed a bit.

#STANDALONE-ALBANY

#-- Generation of ascii files using matlab.

#move to mFiles directory
cd mFiles

# modify (if needed) maltab script "build_cism_msh_from_nc" to fix input/output paths and filenames.
# run matlab script "build_cism_msh_from_nc"
matlab -nojvm < build_cism_msh_from_nc.m

#move back to top directory
cd ..

#create 2d exodus file for Greenland.
#Warning!! this part is very hacky, you'll get a runtime error, but the correct *.exo will be saved in the albanyMesh folder. Also, this can be extremely slow with large files, unless trilinos is compiled with the nodebug option -D CMAKE_CXX_FLAGS:STRING="-O3 -fPIC -fno-var-tracking -DNDEBUG".
mpirun -np 8 Albany inputFiles/create2dExo.yaml


#run standalone Albany simulation
mpirun -np 8 Albany inputFiles/input_standalone-albanyT.yaml 
cd albanyMesh
epu --auto greenland_2d.exo.8.0
cd ..
# [optional] if you run the above on multiple processors, you need to merge the exodus files into one:
#$ path-to-trilinos-install/bin/epu --auto greenland_cism-albanyT.exo.4.

epu --auto greenland_standalone-albanyT.exo.8.0

#COMPARE CISM-ALBANY with STANDALONE ALBANY
#move to mFiles directory
cd mFiles

#run the script compare_exos.m
matlab -nojvm < compare_exos.m

# you'll see the max difference (in absolut value) between fields. Note that the raher significant difference in beta comes from the fact that beta is changed in cism according to the floating condition.

#STORE STANDALONE ALBANY FIELDS INTO nc.
#create a copy of greenland.nc
cp ../ncGridSamples/greenland.nc ../greenland_standalone-albanyT.nc
matlab -nojvm < print_exo_fields_into_nc.m


#Note: When the thickness and the bedrock topography are interpolated back to the grid, some accuracy is lost (try comparing the original "greenland.nc" with the newly created "geenland_standalone-albanyT.nc"). In fact, if you now re-run cism-alabny #using the new nc grid you'll see a significant difference with the standalone albany solution:

#move back to top folder
cd ..

#after modifying the inputFiles/cism-albanyT.config to use the new gid greenland_standalone-albanyT.nc, run cism-albanyT, and compare again
cd inputFiles
mpirun -np 8 ../cism_driver/cism_driver cism-albanyT.config
epu --auto greenland_cism-albanyT.exo.8.0
cd ..

cd mFiles
matlab -nojvm < compare_exos.m

#quite a difference.. this is an interpolation error.. so it should diminish as the grid is refined.

#IKT, WARNING: the following is specific to Irina Tezaur's machine, camobap!
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib64/openmpi/lib:/usr/lib
