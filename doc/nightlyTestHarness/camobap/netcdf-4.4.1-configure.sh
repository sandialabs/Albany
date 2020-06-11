./configure CC=mpicc FC=mpifort CXX=mpicxx \
      CXXFLAGS="-fPIC -I/nightlyCDash/albany-tpls-gcc-10.1.1/include -O3 -march=native" \
      CFLAGS="-fPIC -I/nightlyCDash/albany-tpls-gcc-10.1.1/include -O3 -march=native" \
      LDFLAGS="-fPIC -L/nightlyCDash/albany-tpls-gcc-10.1.1/lib -O3 -march=native" \
      FCFLAGS="-fPIC -I/nightlyCDash/albany-tpls-gcc-10.1.1/include -O3 -march=native" \
      --prefix=/nightlyCDash/albany-tpls-gcc-10.1.1 --disable-doxygen --enable-netcdf4 #--enable-pnetcdf
