./configure CC=mpicc FC=mpifort CXX=mpicxx \
      CXXFLAGS="-fPIC -I/home/ikalash/albany-tpls-gcc-9.3.1/include -O3 -march=native" \
      CFLAGS="-fPIC -I/home/ikalash/albany-tpls-gcc-9.3.1/include -O3 -march=native" \
      LDFLAGS="-fPIC -L/home/ikalash/albany-tpls-gcc-9.3.1/lib -O3 -march=native" \
      FCFLAGS="-fPIC -I/home/ikalash/albany-tpls-gcc-9.3.1/include -O3 -march=native" \
      --prefix=/home/ikalash/albany-tpls-gcc-9.3.1 --disable-doxygen --enable-netcdf4 #--enable-pnetcdf
