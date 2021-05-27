 ./configure CC=mpicc FC=mpifort CXX=mpicxx \
      CXXFLAGS="-fPIC -I/home/ikalash/nightlyAlbanyTests/TPLs -O3 -march=native" \
      CFLAGS="-fPIC -I/home/ikalash/nightlyAlbanyTests/TPLs -O3 -march=native" \
      LDFLAGS="-fPIC -L/projects/albany/lib -O3 -march=native" \
      FCFLAGS="-fPIC -I/home/ikalash/nightlyAlbanyTests/TPLs -O3 -march=native" \
      --prefix=/home/ikalash/nightlyAlbanyTests/TPLs --disable-doxygen --enable-netcdf4 --enable-pnetcdf
