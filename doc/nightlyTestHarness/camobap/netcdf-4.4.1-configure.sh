./configure CC=/usr/lib64/openmpi/bin/mpicc FC=/usr/lib64/openmpi/bin/mpifort CXX=/usr/lib64/openmpi/bin/mpicxx \
      CXXFLAGS="-fPIC -O3 -march=native" \
      CFLAGS="-fPIC -O3 -march=native" \
      LDFLAGS="-fPIC -O3 -march=native" \
      FCFLAGS="-fPIC -O3 -march=native" \
      --prefix=/nightlyCDash/albany-tpls-gcc-11.1.1-openmpi-4.1.0 --disable-doxygen --disable-netcdf4 #--enable-netcdf4 #--enable-pnetcdf
