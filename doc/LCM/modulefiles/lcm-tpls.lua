whatis("LCM TPLs for Trilinos/Albany")

if (isloaded("lcm-intel")) then
  load("openmpi-intel/2.0")
  netcdf_root = "/ascldap/users/daibane/LCM/TPL_toss3/install/netcdf"
end
if (isloaded("lcm-gcc")) then
  load("seacas")
  load("openmpi-gnu/2.0")
  netcdf_root = "/projects/seacas/cts1/current/"
end
local boost_inc = "/ascldap/users/daibane/LCM/TPL_toss3/install/boost/include"
local boost_lib = "/ascldap/users/daibane/LCM/TPL_toss3/install/boost/lib"
local mpi_root = os.getenv("MPI_ROOT")
local netcdf_inc = netcdf_root .. "/include"
local netcdf_lib = netcdf_root .. "/lib"
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
-- PATHs for MPI are set by the modules
prepend_path("LD_LIBRARY_PATH", netcdf_lib)
