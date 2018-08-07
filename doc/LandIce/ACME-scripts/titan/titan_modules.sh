
#module swap PrgEnv-pgi PrgEnv-gnu
#module swap pgi pgi/14.2.0
module load boost cray-netcdf-hdf5parallel cmake/2.8.11.2
# netcdf-hdf5parallel cray-hdf5-parallel/1.8.12 gcc

module list


# ADDED TO LINK LINE: /opt/gcc/4.8.2/snos/lib/libstdc++.a
# WRONG STDC++ FROM HERE:  /usr/lib64/gcc/x86_64-suse-linux/4.3/libstdc++.a
