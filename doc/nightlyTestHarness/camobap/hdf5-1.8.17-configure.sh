./configure --prefix=/nightlyCDash/albany-tpls-gcc-11.1.1-openmpi-4.1.0 --enable-parallel CC=/tpls/install/bin/mpicc FC=/tpls/install/bin/mpifort CXX=/tpls/install/bin/mpicxx \
  CXXFLAGS="-fPIC -O3 -march=native" CFLAGS="-fPIC -O3 -march=native" \
  F77FLAGS="-fPIC -O3 -march=native" F90FLAGS="-fPIC -O3 -march=native"
