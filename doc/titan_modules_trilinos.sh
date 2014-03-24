#!\bin\bash -l
module swap PrgEnv-pgi PrgEnv-gnu   
#module unload cray-hdf5-parallel
module load netcdf-hdf5parallel git cmake/2.8.6 boost
module list
