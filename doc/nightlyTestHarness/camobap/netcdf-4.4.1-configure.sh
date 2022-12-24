
H5DIR=/nightlyCDash/albany-tpls-gcc-11.1.1-openmpi-4.1.0
./configure CC=/tpls/install/bin/mpicc FC=/tpls/install/bin/mpifort CXX=/tpls/install/bin/mpicxx \
      CXXFLAGS="-fPIC -O3 -march=native -I${H5DIR}/include" \
      CPPFLAGS="-fPIC -O3 -march=native -I${H5DIR}/include" \
      CFLAGS="-fPIC -O3 -march=native -I${H5DIR}/include" \
      LDFLAGS="-fPIC -O3 -march=native -I${H5DIR}/include" \
      FCFLAGS="-fPIC -O3 -march=native -I${H5DIR}/include" \
      LDFLAGS="-L${H5DIR}/lib" \
      --prefix=$H5DIR --disable-doxygen --disable-netcdf4 #--enable-netcdf4 #--enable-pnetcdf
