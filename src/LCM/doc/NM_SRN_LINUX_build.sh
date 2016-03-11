#!/bin/bash -l

# set these!

# root dir of the installation
HOME_DIR=/scratch/jtostie/LCM_TEST
echo " HOME_DIR set to $HOME_DIR " 

# number of processes
echo " input the number of processors: "
read N

# set the proxy server
export http_proxy=wwwproxy.sandia.gov:80

echo " http_proxy set to $http_proxy "


##
## Ignore stuff below here
##

# create the Software directory
cd $HOME_DIR
if [ ! -d "$HOME_DIR/Software" ]; then
    mkdir Software
fi
cd Software
SOFT_DIR=$HOME_DIR/Software
echo " SOFT_DIR set to $SOFT_DIR " 

# create the install directory
INSTALL_DIR=$SOFT_DIR/install
echo " INSTALL_DIR set to $INSTALL_DIR " 

echo "update LD_LIBRARY_PATH for soon to be installed libs"
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_DIR/lib64:$LD_LIBRARY_PATH

# get gmp
TARGET=gmp-5.0.2
cd $SOFT_DIR
if [ ! -d "$SOFT_DIR/$TARGET" ]; then
    echo ""
    echo " Fetching $TARGET "
    echo ""
    if [ ! -e "$TARGET.tar.bz2 " ]; then
	curl -O http://ftp.gnu.org/gnu/gmp/$TARGET.tar.bz2
    fi
    echo " -- unpacking $TARGET"
    tar xjf $TARGET.tar.bz2
    cd $TARGET
    echo " -- configuring $TARGET"
    ./configure \
        --prefix=$INSTALL_DIR \
        ABI=64 \
        &> config_gmp.log
    echo " -- building $TARGET"
    make -j $N &> make_gmp.log
    echo " -- checking $TARGET"
    make check &> check_gmp.log
    echo " -- installing $TARGET"
    make install &> install_gmp.log;
fi

# get mpfr
TARGET=mpfr-3.1.0
cd $SOFT_DIR
if [ ! -d "$SOFT_DIR/$TARGET" ]; then
    echo ""
    echo " Fetching $TARGET "
    echo ""
    if [ ! -e "$TARGET.tar.bz2" ]; then
	curl -O http://www.mpfr.org/mpfr-current/$TARGET.tar.bz2
    fi
    echo " -- unpacking $TARGET"
    tar xjf $TARGET.tar.bz2
    cd $TARGET
    echo " -- configuring $TARGET"
    ./configure --prefix=$INSTALL_DIR \
                --with-gmp=$INSTALL_DIR &> config_mpfr.log
    echo " -- building $TARGET"
    make -j $N &> make_mpfr.log
    echo " -- checking $TARGET"
    make check &> check_mpfr.log
    echo " -- installing $TARGET"
    make install &> install_mpfr.log
fi

# get mpc
TARGET=mpc-0.9
cd $SOFT_DIR
if [ ! -d "$SOFT_DIR/$TARGET" ]; then
    echo ""
    echo " Fetching $TARGET "
    echo ""
    if [ ! -e "$TARGET.tar.gz" ]; then
	curl -O http://www.multiprecision.org/mpc/download/$TARGET.tar.gz
    fi
    echo " -- unpacking $TARGET"
    tar xzf $TARGET.tar.gz
    cd $TARGET
    echo " -- configuring $TARGET"
    ./configure --prefix=$INSTALL_DIR \
                --with-gmp=$INSTALL_DIR \
                --with-mpfr=$INSTALL_DIR &> config_mpc.log
    echo " -- building $TARGET"
    make -j $N &> make_mpc.log
    echo " -- checking $TARGET" 
    make check &> check_mpc.log
    echo " -- installing $TARGET" 
    make install &> install_mpc.log
fi

# get gcc
TARGET=gcc-4.6.1
cd $SOFT_DIR
if [ ! -d "$SOFT_DIR/$TARGET" ]; then
    echo ""
    echo " Fetching $TARGET "
    echo ""
    if [ ! -e "$TARGET.tar.bz2" ]; then
	curl -O http://ftp.gnu.org/gnu/gcc/$TARGET/$TARGET.tar.bz2
    fi
    echo " -- unpacking $TARGET"
    tar xjf $TARGET.tar.bz2
    cd $TARGET
    echo " -- configuring $TARGET"
    ./configure --prefix=$INSTALL_DIR \
                --with-gmp=$INSTALL_DIR \
                --with-mpfr=$INSTALL_DIR\
	        --with-mpc=$INSTALL_DIR \
                --disable-multilib &> config_gcc.log
    echo " -- building $TARGET"
    echo " -- NOTE: building gcc takes a long, long time"
    make -j $N &> make_gcc.log
    echo " -- installing $TARGET"
    make install &> install_gcc.log
fi

echo ""
echo " Set compiler variables to use $TARGET"
echo ""
export cc=$INSTALL_DIR/bin/gcc
export CXX=$INSTALL_DIR/bin/g++
export FC=$INSTALL_DIR/bin/gfortran
export F77=$INSTALL_DIR/bin/gfortran

# get openmpi
TARGET=openmpi-1.4.3
cd $SOFT_DIR
if [ ! -d "$SOFT_DIR/$TARGET" ]; then
    echo ""
    echo " Fetching $TARGET "
    echo ""
    if [ ! -e "$TARGET.tar.bz2" ]; then
	curl -O http://www.open-mpi.org/software/ompi/v1.4/downloads/$TARGET.tar.bz2
    fi
    echo " -- unpacking $TARGET"
    tar xjf $TARGET.tar.bz2
    cd $TARGET
    echo " -- configuring $TARGET"
    ./configure --prefix=$INSTALL_DIR &> config_openmppi.log
    echo " -- building $TARGET"
    make -j $N &> make_openmpi.log
    echo " -- installing $TARGET"
    make install &> install_openmpi.log
fi

# get boost
TARGET=boost_1_45_0
cd $SOFT_DIR
if [ ! -d "$SOFT_DIR/$TARGET" ]; then
    echo ""
    echo " Fetching $TARGET "
    echo ""
    if [ ! -e "$TARGET.tar.bz2" ]; then
	curl -L  http://downloads.sourceforge.net/project/boost/boost/1.45.0/$TARGET.tar.bz2 > $TARGET.tar.bz2
    fi
    echo " -- unpacking $TARGET"
    tar xjf $TARGET.tar.bz2
    cd $TARGET
    echo " -- configuring $TARGET"
    ./bootstrap.sh --prefix=$INSTALL_DIR &> config_boost.log
    echo " -- installing $TARGET"
    echo " -- NOTE: installing boost takes a long, long time"
    ./bjam install -j $N &> install_boost.log
fi

# get netcdf
TARGET=netcdf-4.1.1
cd $SOFT_DIR
if [ ! -d "$SOFT_DIR/$TARGET" ]; then
    echo ""
    echo " Fetching $TARGET "
    echo ""
    if [ ! -e "$TARGET.tar.gz" ]; then
	curl -O http://www.unidata.ucar.edu/downloads/netcdf/ftp/$TARGET.tar.gz
    fi
    echo " -- unpacking $TARGET"
    tar xzf $TARGET.tar.gz
    cd $TARGET
    echo " -- configuring $TARGET"
    ./configure --prefix=$INSTALL_DIR \
	        --disable-netcdf-4 \
	        --disable-dap  &> config_netcdf.log
    echo " -- building $TARGET"
    make -j $N &> make_netcdf.log
    echo " -- checking $TARGET"
    make check &> check_netcdf.log
    echo " -- installing $TARGET"
    make install &> install_netcdf.log
fi

# get CMAKE
TARGET=cmake-2.8.5
cd $SOFT_DIR
if [ ! -d "$SOFT_DIR/$TARGET" ]; then
    echo ""
    echo " Fetching $TARGET "
    echo ""
    if [ ! -e "$TARGET.tar.gz" ]; then
	curl -O http://www.cmake.org/files/v2.8/cmake-2.8.5.tar.gz
    fi
    echo " -- unpacking $TARGET"
    tar xzf $TARGET.tar.gz
    cd $TARGET
    echo " -- configuring $TARGET"
    ./bootstrap --prefix=$INSTALL_DIR &> config_cmake.log
    echo " -- building $TARGET"
    make -j $N &> make_cmake.log
    echo " -- installing $TARGET"
    make install &> install_cmake.log
fi

# get eg
TARGET=eg
cd $INSTALL_DIR
if [ ! -d "bin" ]; then
    mkdir bin
fi
cd bin
if [ ! -e "eg" ]; then
    echo ""
    echo " Fetching $TARGET "
    echo ""
    curl -O http://people.gnome.org/~newren/eg/download/1.7.3/eg
    chmod a+x eg
fi

export PATH=$INSTALL_DIR/bin:$PATH

echo "**** these need to be added to your path for future use ****"
echo ""
echo " export LD_LIBRARY_PATH=$INSTALL_DIR/lib:\$LD_LIBRARY_PATH"
echo " export LD_LIBRARY_PATH=$INSTALL_DIR/lib64:\$LD_LIBRARY_PATH"
echo " export PATH=$INSTALL_DIR/bin:\$PATH"
echo ""

cd $HOME_DIR
echo " 
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:\$LD_LIBRARY_PATH \\
export LD_LIBRARY_PATH=$INSTALL_DIR/lib64:\$LD_LIBRARY_PATH \\
export PATH=$INSTALL_DIR/bin:\$PATH" > LCM.conf

source LCM.conf

echo " Moving on to LCM installation "
# Set up LCM dirs
cd $HOME_DIR
if [ ! -d "LCM" ]; then
    mkdir LCM
fi
LCM_DIR=$HOME_DIR/LCM
echo " LCM_DIR is $LCM_DIR"

# get Trilinos
TARGET=TRILINOS
cd $LCM_DIR
if [ ! -d "Trilinos" ]; then
    echo ""
    echo " Fetching $TARGET "
    echo ""
    eg clone software.sandia.gov:/space/git/Trilinos
fi
cd Trilinos
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
echo "#!/bin/bash
rm CMakeCache.txt
cmake  \\
    -D CMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR \\
    -D CMAKE_BUILD_TYPE:STRING=NONE \\
    -D Trilinos_WARNINGS_AS_ERRORS_FLAGS:STRING="" \\
    -D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \\
    -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF \\
    -D TPL_ENABLE_MPI:BOOL=ON \\
    -D MPI_BASE_DIR:PATH=$INSTALL_DIR \\
    -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \\
    -D Trilinos_VERBOSE_CONFIGURE:BOOL=OFF \\
    -D Trilinos_ENABLE_DEBUG:BOOL=OFF \\
    -D CMAKE_CXX_FLAGS:STRING="-O2 -g -ansi" \\
    -D TPL_ENABLE_Boost:BOOL=ON \\
    -D Boost_INCLUDE_DIRS:PATH=$INSTALL_DIR/include \\
    -D Trilinos_ENABLE_Teuchos:BOOL=ON \\
    -D Trilinos_ENABLE_Shards:BOOL=ON \\
    -D Trilinos_ENABLE_Sacado:BOOL=ON \\
    -D Trilinos_ENABLE_Epetra:BOOL=ON \\
    -D Trilinos_ENABLE_EpetraExt:BOOL=ON \\
    -D EpetraExt_USING_HDF5:BOOL=OFF \\
    -D Trilinos_ENABLE_Ifpack:BOOL=ON \\
    -D Trilinos_ENABLE_AztecOO:BOOL=ON \\
    -D Trilinos_ENABLE_Amesos:BOOL=ON \\
    -D Trilinos_ENABLE_Anasazi:BOOL=ON \\
    -D Trilinos_ENABLE_Belos:BOOL=ON \\
    -D Trilinos_ENABLE_ML:BOOL=ON \\
    -D Trilinos_ENABLE_Phalanx:BOOL=ON \\
    -D Trilinos_ENABLE_Intrepid2:BOOL=ON \\
    -D Trilinos_ENABLE_NOX:BOOL=ON \\
    -D Trilinos_ENABLE_Stratimikos:BOOL=ON \\
    -D Trilinos_ENABLE_Thyra:BOOL=ON \\
    -D Trilinos_ENABLE_Rythmos:BOOL=ON \\
    -D Trilinos_ENABLE_MOOCHO:BOOL=ON \\
    -D Trilinos_ENABLE_Stokhos:BOOL=ON \\
    -D Trilinos_ENABLE_Piro:BOOL=ON \\
    -D Trilinos_ENABLE_STK:BOOL=ON \\
    -D Trilinos_ENABLE_Teko:BOOL=ON \\
    -D Trilinos_ENABLE_SEACASIoss:BOOL=ON \\
    -D Trilinos_ENABLE_Pamgen:BOOL=ON \\
    -D Trilinos_ENABLE_Zoltan:BOOL=ON \\
    -D Trilinos_ENABLE_ThreadPool:BOOL=ON \\
    -D TPL_ENABLE_BinUtils=OFF \\
    -D TPL_ENABLE_Netcdf:BOOL=ON \\
    -D Netcdf_INCLUDE_DIRS:PATH=$INSTALL_DIR/include \\
    -D Netcdf_LIBRARY_DIRS:PATH=$INSTALL_DIR/lib \\
    -D TPL_ENABLE_HDF5:BOOL=OFF \\
    -D Trilinos_ENABLE_TESTS:BOOL=OFF \\
    -D Trilinos_ENABLE_EXAMPLES:BOOL=OFF \\
    -D Teuchos_ENABLE_TESTS:BOOL=ON \\
    ../" > trilinos_config.sh
chmod a+x trilinos_config.sh
echo " -- configuring $TARGET"
./trilinos_config.sh &> config_tri.log
echo " -- building $TARGET"
make install -j $N &> make_tri.log

# We already have Albany
TARGET=ALBANY
cd $LCM_DIR/Albany
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
echo "#!/bin/bash
rm CMakeCache.txt
cmake  \\
    -D ALBANY_TRILINOS_DIR:FILEPATH=$INSTALL_DIR \\
    -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \\
    -D ENABLE_LCM:BOOL=ON \\
    ../" > albany_config.sh
chmod a+x albany_config.sh
echo " -- configuring $TARGET"
./albany_config.sh &> config_alb.log
echo " -- building $TARGET"
make -j $N &> make_alb.log
echo " -- testing $TARGET"
ctest 

