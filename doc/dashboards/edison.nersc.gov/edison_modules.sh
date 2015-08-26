
# gcc module needed for C++11 constructs. Intel is still
# the compiler, but gcc headers are needed. Got this from
# Edison help page.

# cray-hdf5-parallel defaults to 1.8.13 which is incompatible with netcdf 
# -- need to revert to 1.8.12 -- found this by googling unreolved linking errors

module load boost cmake netcdf-hdf5parallel cray-hdf5-parallel/1.8.12 gcc

module list
