whatis("LCM TPLs for Trilinos/Albany")

load("openmpi-intel/2.0")
local boost_inc = "/ascldap/users/daibane/LCM/TPL_toss3/install/boost/include"
local boost_lib = "/ascldap/users/daibane/LCM/TPL_toss3/install/boost/lib"
local mpi_root = os.getenv("MPI_ROOT")
local netcdf_root = "/ascldap/users/daibane/LCM/TPL_toss3/install/netcdf"
local netcdf_inc = netcdf_root .. "/include"
local netcdf_lib = netcdf_root .. "/lib"
local yaml_cpp_inc = "/ascldap/users/daibane/LCM/TPL_toss3/install/yaml-cpp/include"
local yaml_cpp_lib = "/ascldap/users/daibane/LCM/TPL_toss3/install/yaml-cpp/lib"
setenv("BOOST_INC", boost_inc)
setenv("BOOST_LIB", boost_lib)
setenv("BOOSTLIB_INC", boost_inc)
setenv("BOOSTLIB_LIB", boost_lib)
setenv("MPI_INC", mpi_root .. "/include")
setenv("MPI_LIB", mpi_root .. "/lib")
setenv("MPI_BIN", mpi_root .. "/bin")
setenv("NETCDF", netcdf_root)
setenv("NETCDF_INC", netcdf_inc)
setenv("NETCDF_LIB", netcdf_lib)
setenv("LCM_NETCDF_PARALLEL", "ON")
setenv("YAML_CPP_INC", yaml_cpp_inc)
setenv("YAML_CPP_LIB", yaml_cpp_lib)
-- PATHs for MPI are set by the modules
prepend_path("LD_LIBRARY_PATH", netcdf_lib)
