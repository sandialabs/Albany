
# gcc module needed for C++11 constructs. Intel is still
# the compiler, but gcc headers are needed. Got this from
# Edison help page.

module load boost cmake gcc

# cray-hdf5-parallel defaults to 1.8.13 which is incompatible with netcdf 
# -- need to revert to 1.8.12 -- found this by googling unreolved linking errors

# netcdf/hdf5 conflict with ACME versions, and not needed for ACME builds

# Following needed for stand-alone albany builds, but not for ACME
#modile load netcdf-hdf5parallel cray-hdf5-parallel/1.8.12

module list
