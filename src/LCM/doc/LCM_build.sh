#!/bin/bash -l

# set these!

# root dir of the installation
HOME_DIR=/scratch/jtostie
echo " HOME_DIR set to $HOME_DIR " 

# if you are on the NM LAN uncomment this
# answer yes to the question
# !>> begin here
export http_proxy=wwwproxy.sandia.gov:80
module clear
module load sierra-git
module load sierra-cmake
echo ""
module list
echo ""
# !<< end here

# if you are in CA uncomment this
#export http_proxy=wwwproxy.ran.sandia.gov:80

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
TARGET=GMP
cd $SOFT_DIR
if [ ! -d "$SOFT_DIR/gmp-5.0.1" ]; then
    echo ""
    echo " Fetching $TARGET "
    echo ""
    if [ ! -e "gmp-5.0.1.tar.bz2 " ]; then
	curl -O http://download.uni-hd.de/ftp/pub/gnu/gmp/gmp-5.0.1.tar.bz2
    fi
    echo " -- unpacking $TARGET"
    tar xjf gmp-5.0.1.tar.bz2
    cd gmp-5.0.1
    echo " -- configuring $TARGET"
    ./configure --prefix=$INSTALL_DIR &> config_gmp.log
    echo " -- building $TARGET"
    make -j 8 &> make_gmp.log
    echo " -- checking $TARGET"
    make check &> check_gmp.log
    echo " -- installing $TARGET"
    make install &> install_gmp.log;
fi

# get mpfr
TARGET=MPFR
cd $SOFT_DIR
if [ ! -d "$SOFT_DIR/mpfr-3.0.0" ]; then
    echo ""
    echo " Fetching $TARGET "
    echo ""
    if [ ! -e "mpfr-3.0.0.tar.bz2" ]; then
	curl -O http://www.mpfr.org/mpfr-current/mpfr-3.0.0.tar.bz2
    fi
    echo " -- unpacking $TARGET"
    tar xjf mpfr-3.0.0.tar.bz2
    cd mpfr-3.0.0
    echo " -- configuring $TARGET"
    ./configure --prefix=$INSTALL_DIR \
                --with-gmp=$INSTALL_DIR &> config_mpfr.log
    echo " -- building $TARGET"
    make -j 8 &> make_mpfr.log
    echo " -- checking $TARGET"
    make check &> check_mpfr.log
    echo " -- installing $TARGET"
    make install &> install_mpfr.log
fi

# get mpc-0.8.2
TARGET=MPC
cd $SOFT_DIR
if [ ! -d "$SOFT_DIR/mpc-0.8.2" ]; then
    echo ""
    echo " Fetching $TARGET "
    echo ""
    if [ ! -e "mpc-0.8.2.tar.gz" ]; then
	curl -O http://www.multiprecision.org/mpc/download/mpc-0.8.2.tar.gz
    fi
    echo " -- unpacking $TARGET"
    tar xzf mpc-0.8.2.tar.gz
    cd mpc-0.8.2
    echo " -- configuring $TARGET"
    ./configure --prefix=$INSTALL_DIR \
                --with-gmp=$INSTALL_DIR \
                --with-mpfr=$INSTALL_DIR &> config_mpc.log
    echo " -- building $TARGET"
    make -j 8 &> make_mpc.log
    echo " -- checking $TARGET" 
    make check &> check_mpc.log
    echo " -- installing $TARGET" 
    make install &> install_mpc.log
fi

# get gcc-4.5.1
TARGET=GCC
cd $SOFT_DIR
if [ ! -d "$SOFT_DIR/gcc-4.5.1" ]; then
    echo ""
    echo " Fetching $TARGET "
    echo ""
    if [ ! -e "gcc-4.5.1.tar.bz2" ]; then
	curl -O http://ftp.gnu.org/gnu/gcc/gcc-4.5.1/gcc-4.5.1.tar.bz2
    fi
    echo " -- unpacking $TARGET"
    tar xjf gcc-4.5.1.tar.bz2
    cd gcc-4.5.1
    echo " -- configuring $TARGET"
    ./configure --prefix=$INSTALL_DIR \
                --with-gmp=$INSTALL_DIR \
                --with-mpfr=$INSTALL_DIR\
	        --with-mpc=$INSTALL_DIR \
                --disable-multilib &> config_gcc.log
    echo " -- building $TARGET"
    echo " -- NOTE: building gcc takes a long, long time"
    make -j 8 &> make_gcc.log
    echo " -- installing $TARGET"
    make install &> install_gcc.log
fi

echo  "set compiler variables to use gcc-4.5.1"
export CC=$INSTALL_DIR/bin/gcc
export CXX=$INSTALL_DIR/bin/g++
export FC=$INSTALL_DIR/bin/gfortran
export F77=$INSTALL_DIR/bin/gfortran

# get openmpi-1.4.3
TARGET=OPENMPI
cd $SOFT_DIR
if [ ! -d "$SOFT_DIR/openmpi-1.4.3" ]; then
    echo ""
    echo " Fetching $TARGET "
    echo ""
    if [ ! -e "openmpi-1.4.3.tar.bz2" ]; then
	curl -O http://www.open-mpi.org/software/ompi/v1.4/downloads/openmpi-1.4.3.tar.bz2
    fi
    echo " -- unpacking $TARGET"
    tar xjf openmpi-1.4.3.tar.bz2
    cd openmpi-1.4.3
    echo " -- configuring $TARGET"
    ./configure --prefix=$INSTALL_DIR &> config_openmppi.log
    echo " -- building $TARGET"
    make -j 8 &> make_openmpi.log
    echo " -- installing $TARGET"
    make install &> install_openmpi.log
fi

# get boost 1.45.0
TARGET=BOOST
cd $SOFT_DIR
if [ ! -d "$SOFT_DIR/boost_1_45_0" ]; then
    if [ -e "/home/jtostie/boost_1_45_0.tar.bz2" ]; then
	echo ""
	echo " Fetching $TARGET "
	echo ""
	if [ ! -e "boost_1_45_0.tar.bz2" ]; then
	    cp /home/jtostie/boost_1_45_0.tar.bz2 .
	fi
	echo " -- unpacking $TARGET"
	tar xjf boost_1_45_0.tar.bz2
	cd boost_1_45_0
	echo " -- configuring $TARGET"
	./bootstrap.sh --prefix=/$INSTALL_DIR &> config_boost.log
	echo " -- installing $TARGET"
	echo " -- NOTE: installing boost takes a long, long time"
	./bjam install -j 8 &> install_boost.log
    else
	echo " please download and install boost version 1.45.0 "
    fi
fi

# get netcdf
TARGET=NETCDF
cd $SOFT_DIR
if [ ! -d "$SOFT_DIR/netcdf-4.1.1" ]; then
    echo ""
    echo " Fetching $TARGET "
    echo ""
    if [ ! -e "netcdf-4.1.1.tar.gz" ]; then
	curl -O http://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-4.1.1.tar.gz
    fi
    echo " -- unpacking $TARGET"
    tar xzf netcdf-4.1.1.tar.gz
    cd netcdf-4.1.1
    echo " -- configuring $TARGET"
    ./configure --prefix=$INSTALL_DIR \
	        --disable-netcdf-4 \
	        --disable-dap  &> config_netcdf.log
    echo " -- building $TARGET"
    make -j 8 &> make_netcdf.log
    echo " -- checking $TARGET"
    make check &> check_netcdf.log
    echo " -- installing $TARGET"
    make install &> install_netcdf.log
fi

export PATH=$INSTALL_DIR/bin:$PATH

echo "**** these need to be added to your path for future use ****"
echo ""
echo " export LD_LIBRARY_PATH=$INSTALL_DIR/lib:LD_LIBRARY_PATH"
echo " export LD_LIBRARY_PATH=$INSTALL_DIR/lib64:LD_LIBRARY_PATH"
echo " export PATH=$INSTALL_DIR/bin:PATH"
echo ""


echo " Moving on to LCM installation "
# Set up LCM dirs
cd $SOFT_DIR
if [ ! -d "LCM" ]; then
    mkdir LCM
fi
LCM_DIR=$SOFT_DIR/LCM
echo " LCM_DIR is $LCM_DIR"

# get Trilinos
TARGET=TRILINOS
cd $LCM_DIR
if [ ! -d "Trilinos" ]; then
    echo ""
    echo " Fetching $TARGET "
    echo ""
    eg clone software.sandia.gov:/space/git/Trilinos
    cd Trilinos
    mkdir build
    cd build
    if [ ! -e "$HOME_DIR/trilinos_config" ]; then
	echo "!! please put your trilinos_config file in $HOME_DIR so that I can configure Trilinos"
	exit
    else
	cp $HOME_DIR/trilinos_config $LCM_DIR/Trilinos/build
    fi
    echo " -- configuring $TARGET"
    ./trilinos_config &> config_tri.log
    echo " -- building $TARGET"
    make install -j 8 &> make_tri.log
    echo " -- testing $TARGET"
    ctest
fi

# get Albany
TARGET=ALBANY
cd $LCM_DIR
if [ ! -d "Albany" ]; then
    echo ""
    echo " Fetching $TARGET "
    echo ""
    eg clone software.sandia.gov:/space/git/Albany
    cd Albany
    mkdir build
    cd build
    if [ ! -e "$HOME_DIR/albany_config" ]; then
	echo "!! please put your albany_config file in $HOME_DIR so that I can configure Albany"
	exit
    else
	cp $HOME_DIR/albany_config $LCM_DIR/Albany/build
    fi
    echo " -- configuring $TARGET"
    ./albany_config &> config_alb.log
    echo " -- building $TARGET"
    make -j 8 &> make_alb.log
    echo " -- testing $TARGET"
    ctest 
fi
