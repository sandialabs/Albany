
#!/bin/bash

#IKT, WARNING: the following 3 lines are specific to Irina Tezaur's machine, mockba!
#They need to be changed for other machines! 
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:/projects/sems/install/rhel7-x86_64/sems/v2/tpl/netcdf-fortran/4.5.3/gcc/10.1.0/openmpi/1.10.7/i7xst5r/lib:/projects/sems/install/rhel7-x86_64/sems/v2/tpl/netcdf-c/4.7.3/gcc/10.1.0/openmpi/1.10.7/oyw32kr/lib:/projects/sems/install/rhel7-x86_64/sems/v2/tpl/parallel-netcdf/1.12.1/gcc/10.1.0/openmpi/1.10.7/jlvymqv/lib:/projects/sems/install/rhel7-x86_64/sems/v2/tpl/hdf5/1.10.7/gcc/10.1.0/openmpi/1.10.7/uubef2f/lib
export PATH=$PATH:/usr/local/usr/local/MATLAB/R2022b/bin:/home/ikalash/Trilinos_Albany/home/ikalash/Trilinos_Albany/nightlyAlbanyTests/Results/Trilinos/build/install/bin

rm -rf *exo*
rm -rf albanyMesh/*exo*

# CISM-ALBANY

# run cism-albany after modifying (if needed) the paths of the input nc "name" file and the "dycore_input_file" in the file inputFiles/cism-albanyT.config.
cd inputFiles
rm -rf *exo* 
/projects/sems/install/rhel7-x86_64/sems/v2/tpl/openmpi/1.10.7/gcc/10.1.0/base/7jgrwmo/bin/mpiexec -np 8 ../cism_driver/cism_driver cism-albanyT.config
/home/ikalash/Trilinos_Albany/nightlyAlbanyTests/Results/Trilinos/build/install/bin/epu --auto greenland_cism-albanyT.exo.8.0
cd ..

# [optional] if you run the above on multiple processors, you need to merge the exodus files into one:
#/home/ikalash/Trilinos_Albany/nightlyAlbanyTests/Results/Trilinos/build/install/bin/epu --auto greenland_cism-albanyT.exo.4.

#note that if you diff the original greenland.nc file and the one stored by cism greenland_cism-albanyT.nc, beta changed a bit.

#STANDALONE-ALBANY

#-- Generation of ascii files using matlab.

#move to mFiles directory
cd mFiles

# modify (if needed) maltab script "build_cism_msh_from_nc" to fix input/output paths and filenames.
# run matlab script "build_cism_msh_from_nc"
/usr/local/MATLAB/R2022b/bin/matlab -batch "build_cism_msh_from_nc; settings; exit"

#move back to top directory
cd ..

#create 2d exodus file for Greenland.
#Warning!! this part is very hacky, you'll get a runtime error, but the correct *.exo will be saved in the albanyMesh folder. Also, this can be extremely slow with large files, unless trilinos is compiled with the nodebug option -D CMAKE_CXX_FLAGS:STRING="-O3 -fPIC -fno-var-tracking -DNDEBUG".
/projects/sems/install/rhel7-x86_64/sems/v2/tpl/openmpi/1.10.7/gcc/10.1.0/base/7jgrwmo/bin/mpiexec -np 8 Albany inputFiles/create2dExo.yaml


#run standalone Albany simulation
/projects/sems/install/rhel7-x86_64/sems/v2/tpl/openmpi/1.10.7/gcc/10.1.0/base/7jgrwmo/bin/mpiexec -np 8 Albany inputFiles/input_standalone-albanyT.yaml 
cd albanyMesh
/home/ikalash/Trilinos_Albany/nightlyAlbanyTests/Results/Trilinos/build/install/bin/epu --auto greenland_2d.exo.8.0
cd ..
# [optional] if you run the above on multiple processors, you need to merge the exodus files into one:
#$ path-to-trilinos-install/bin//home/ikalash/Trilinos_Albany/nightlyAlbanyTests/Results/Trilinos/build/install/bin/epu --auto greenland_cism-albanyT.exo.4.

/home/ikalash/Trilinos_Albany/nightlyAlbanyTests/Results/Trilinos/build/install/bin/epu --auto greenland_standalone-albanyT.exo.8.0

#COMPARE CISM-ALBANY with STANDALONE ALBANY
#move to mFiles directory
cd mFiles

#run the script compare_exos.m
/usr/local/MATLAB/R2022b/bin/matlab -batch "compare_exos; settings; exit"

# you'll see the max difference (in absolut value) between fields. Note that the raher significant difference in beta comes from the fact that beta is changed in cism according to the floating condition.

#STORE STANDALONE ALBANY FIELDS INTO nc.
#create a copy of greenland.nc
cp ../ncGridSamples/greenland.nc ../greenland_standalone-albanyT.nc
/usr/local/MATLAB/R2022b/bin/matlab -batch "print_exo_fields_into_nc; settings; exit"


#Note: When the thickness and the bedrock topography are interpolated back to the grid, some accuracy is lost (try comparing the original "greenland.nc" with the newly created "geenland_standalone-albanyT.nc"). In fact, if you now re-run cism-alabny #using the new nc grid you'll see a significant difference with the standalone albany solution:

#move back to top folder
cd ..

#after modifying the inputFiles/cism-albanyT.config to use the new gid greenland_standalone-albanyT.nc, run cism-albanyT, and compare again
cd inputFiles
/projects/sems/install/rhel7-x86_64/sems/v2/tpl/openmpi/1.10.7/gcc/10.1.0/base/7jgrwmo/bin/mpiexec -np 8 ../cism_driver/cism_driver cism-albanyT.config
/home/ikalash/Trilinos_Albany/nightlyAlbanyTests/Results/Trilinos/build/install/bin/epu --auto greenland_cism-albanyT.exo.8.0
cd ..

cd mFiles
/usr/local/MATLAB/R2022b/bin/matlab -batch "compare_exos; settings; exit"

#quite a difference.. this is an interpolation error.. so it should diminish as the grid is refined.

