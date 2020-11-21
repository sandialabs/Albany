
export FCFLAGS="-w -fallow-argument-mismatch -O2"
export FFLAGS="-w -fallow-argument-mismatch -O2"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nightlyCDash/albany-tpls-gcc-10.2.1/lib
  
./configure --disable-fortran-type-check --prefix=/nightlyCDash/albany-tpls-gcc-10.2.1 CPPFLAGS=-I/nightlyCDash/albany-tpls-gcc-10.2.1/include LDFLAGS=-L/nightlyCDash/albany-tpls-gcc-10.2.1/lib
