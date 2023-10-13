
INSTALLDIR=/nightlyCDash/albany-tpls-gcc-11.1.1-openmpi-4.1.0
./configure CC=/tpls/install/bin/mpicc FC=/tpls/install/bin/mpifort CXX=/tpls/install/bin/mpicxx \
      CXXFLAGS="-fPIC -O3 -march=native -I${INSTALLDIR}/include" \
      CPPFLAGS="-fPIC -O3 -march=native -I${INSTALLDIR}/include" \
      CFLAGS="-fPIC -O3 -march=native -I${INSTALLDIR}/include" \
      LDFLAGS="-fPIC -O3 -march=native -I${INSTALLDIR}/include" \
      FCFLAGS="-fPIC -O3 -march=native -I${INSTALLDIR}/include" \
      LDFLAGS="-L${INSTALLDIR}/lib" \
      --prefix=$INSTALLDIR 
#--disable-doxygen --disable-netcdf4 #--enable-netcdf4 #--enable-pnetcdf
