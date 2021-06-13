
export FCFLAGS="-w -fallow-argument-mismatch -O2"
export FFLAGS="-w -fallow-argument-mismatch -O2"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nightlyCDash/albany-tpls-gcc-11.1.1-openmpi-4.1.0/lib
  
./configure --disable-fortran-type-check --prefix=/nightlyCDash/albany-tpls-gcc-11.1.1-openmpi-4.1.0 CPPFLAGS=-I/nightlyCDash/albany-tpls-gcc-11.1.1-openmpi-4.1.0/include LDFLAGS=-L/nightlyCDash/albany-tpls-gcc-11.1.1-openmpi-4.1.0/lib
