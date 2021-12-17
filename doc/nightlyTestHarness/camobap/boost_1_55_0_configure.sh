echo "using mpi : /usr/lib64/openmpi/bin/mpicxx ;" >> ./tools/build/v2/user-config.jam
#  echo "using gcc : /usr/bin/g++ ;" >> ./tools/build/v2/user-config.jam
  ./bootstrap.sh --with-libraries=signals,regex,filesystem,system,mpi,serialization,thread,program_options,exception --prefix=/home/ikalash/nightlyAlbanyCDash/tpls-install
