echo "using mpi : /nightlyAlbanyTests/spack/opt/spack/linux-rhel8-haswell/gcc-11.1.0/openmpi-4.1.2-w65ftkfoqlafeb3nlcngwixdd4dwmkof/bin/mpicxx ;" >> ./tools/build/v2/user-config.jam
#  echo "using gcc : /usr/bin/g++ ;" >> ./tools/build/v2/user-config.jam
  ./bootstrap.sh --with-libraries=signals,regex,filesystem,system,mpi,serialization,thread,program_options,exception --prefix=/tpls/install
