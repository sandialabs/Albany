./configure CC=/usr/lib64/openmpi/bin/mpicc FC=/usr/lib64/openmpi/bin/mpifort CXX=/usr/lib64/openmpi/bin/mpicxx \
      CXXFLAGS="-fPIC -I/nightlyCDash/albany-tpls-gcc-10.2.1/include -O3 -march=native" \
      CFLAGS="-fPIC -I/nightlyCDash/albany-tpls-gcc-10.2.1/include -O3 -march=native" \
      LDFLAGS="-fPIC -L/nightlyCDash/albany-tpls-gcc-10.2.1/lib -O3 -march=native" \
      FCFLAGS="-fPIC -I/nightlyCDash/albany-tpls-gcc-10.2.1/include -O3 -march=native" \
      --prefix=/nightlyCDash/albany-tpls-gcc-10.2.1 --disable-doxygen --enable-netcdf4 #--enable-pnetcdf
