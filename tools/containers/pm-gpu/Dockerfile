# Container that mimics the Perlmutter GPU (pm-gpu) nightly build environment.
# See doc/dashboards/perlmutter.nersc.gov/cuda for the build being mimicked.
#
#   Perlmutter module                   ->  Container equivalent
#   ---------------------------------------------------------------------
#   cudatoolkit/12.9                    ->  nvidia/cuda:12.9.1-devel base image
#   gcc-native/14                       ->  gcc-14/g++-14/gfortran-14
#   cray-mpich/9.0.1                    ->  MPICH (wrappers pointed at gcc-14)
#   cray-libsci/25.09.0                 ->  OpenBLAS (BLAS + LAPACK)
#   cray-hdf5-parallel/1.14.3.7         ->  HDF5 1.14.3 (parallel, from source)
#   cray-netcdf-hdf5parallel/4.9.2.1    ->  NetCDF-C 4.9.2 (parallel, from source)
#   cray-parallel-netcdf/1.12.3.19      ->  PnetCDF 1.12.3 (from source)
#   cmake/3.30.2                        ->  CMake 3.30.2 (Kitware binary)
#   boost 1.83.0                        ->  libboost-dev (1.83 on ubuntu 24.04)
#   nvcc_wrapper_a100                   ->  Kokkos nvcc_wrapper, host compiler mpicxx
#
# There are no GPUs on github runners, so this image can only be used to build.
FROM nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu24.04

LABEL maintainer="jwatkin@sandia.gov"

ENV DEBIAN_FRONTEND=noninteractive

# Install basic utils
RUN apt-get update &&   \
    apt-get install -y  \
        curl            \
        git             \
        locate          \
        vim             \
        wget            \
        m4              \
        zlib1g-dev

# Install compilers and python (gcc-native/14 on Perlmutter)
RUN apt-get update &&   \
    apt-get install -y  \
        build-essential \
        gcc-14          \
        g++-14          \
        gfortran-14     \
        python3

# Install MPI (cray-mpich on Perlmutter). Point the MPICH wrappers at gcc-14.
RUN apt-get update &&   \
    apt-get install -y  \
        mpich           \
        libmpich-dev
ENV MPICH_CC=gcc-14
ENV MPICH_CXX=g++-14
ENV MPICH_FC=gfortran-14
ENV MPICH_F77=gfortran-14
ENV CC=mpicc
ENV CXX=mpicxx
ENV FC=mpifort

# Install blas/lapack (cray-libsci on Perlmutter), boost, yaml-cpp, gtest
RUN apt-get update &&   \
    apt-get install -y  \
        libyaml-cpp-dev \
        libboost-dev    \
        libopenblas-dev \
        libgtest-dev && \
    apt-get clean

# Install cmake 3.30.2 (cmake/3.30.2 on Perlmutter)
ARG CMAKE_VERSION=3.30.2
RUN cd /tmp && \
    wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \
    tar -xzf cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz -C /usr/local --strip-components=1 && \
    rm cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz

# Create a bash goodies file, for users to load when manually running the container
# Simply run 'source /opt/share/bash-goodies.sh' to get these
RUN mkdir -p /opt/share && \
    echo 'bind "\"\\e[A\": history-search-backward"' > /opt/share/bash-goodies.sh && \
    echo 'bind "\"\\e[B\": history-search-forward"' >> /opt/share/bash-goodies.sh && \
    echo 'git config --global url."https://github.com/".insteadOf git@github.com:' >> /opt/share/bash-goodies.sh

# Add a runner user, and assign it to the fanssie group
RUN groupadd fanssie && \
    useradd -m -g fanssie -s /bin/bash runner && \
    echo "umask 002" >> /home/runner/.bashrc

# Install sudo, set password for 'runner', and configure sudoers.
# This helps if we realize we need an extra pkg while running a container
RUN apt-get update && \
    apt-get install -y sudo && \
    echo "runner:fanssie" | chpasswd && \
    echo "runner ALL=(ALL) ALL" >> /etc/sudoers && \
    apt-get clean

ENV SOFTWARE_ROOT=/software
RUN mkdir $SOFTWARE_ROOT && \
    chown :fanssie $SOFTWARE_ROOT && \
    chmod 2770 $SOFTWARE_ROOT

# Switch to user from now on
USER runner

# I/O TPLs. Ubuntu's parallel netcdf/pnetcdf packages are built against OpenMPI,
# so build them (and hdf5) from source against MPICH, with Perlmutter's versions.
ENV TPLS_ROOT=${SOFTWARE_ROOT}/tpls
ENV HDF5_DIR=${TPLS_ROOT}
ENV NETCDF_DIR=${TPLS_ROOT}
ENV PNETCDF_DIR=${TPLS_ROOT}
ENV PATH=${TPLS_ROOT}/bin:${PATH}
ENV LD_LIBRARY_PATH=${TPLS_ROOT}/lib:${LD_LIBRARY_PATH}
ENV HDF5_USE_FILE_LOCKING=FALSE

ARG HDF5_VERSION=1.14.3
ARG PNETCDF_VERSION=1.12.3
ARG NETCDF_VERSION=4.9.2

# HDF5 (cray-hdf5-parallel). Note: the release tarball unpacks into ./hdfsrc
RUN mkdir -p ${TPLS_ROOT}/src && cd ${TPLS_ROOT}/src && \
    HDF5_TAG=hdf5-$(echo ${HDF5_VERSION} | tr . _) && \
    wget -q https://github.com/HDFGroup/hdf5/releases/download/${HDF5_TAG}/${HDF5_TAG}.tar.gz -O hdf5.tar.gz && \
    mkdir hdf5 && tar -xzf hdf5.tar.gz -C hdf5 --strip-components=2 && \
    cd hdf5 && \
    ./configure --prefix=${TPLS_ROOT} --enable-parallel --enable-shared --disable-static \
                --enable-build-mode=production --disable-tests --disable-tools && \
    make -j8 install && \
    cd ${TPLS_ROOT}/src && rm -rf hdf5 hdf5.tar.gz

# PnetCDF (cray-parallel-netcdf)
RUN cd ${TPLS_ROOT}/src && \
    wget -q https://parallel-netcdf.github.io/Release/pnetcdf-${PNETCDF_VERSION}.tar.gz -O pnetcdf.tar.gz && \
    mkdir pnetcdf && tar -xzf pnetcdf.tar.gz -C pnetcdf --strip-components=1 && \
    cd pnetcdf && \
    ./configure --prefix=${TPLS_ROOT} --enable-shared --disable-static --disable-cxx \
                MPICC=mpicc MPICXX=mpicxx MPIF77=mpifort MPIF90=mpifort && \
    make -j8 install && \
    cd ${TPLS_ROOT}/src && rm -rf pnetcdf pnetcdf.tar.gz

# NetCDF-C (cray-netcdf-hdf5parallel)
RUN cd ${TPLS_ROOT}/src && \
    wget -q https://downloads.unidata.ucar.edu/netcdf-c/${NETCDF_VERSION}/netcdf-c-${NETCDF_VERSION}.tar.gz -O netcdf.tar.gz && \
    mkdir netcdf && tar -xzf netcdf.tar.gz -C netcdf --strip-components=1 && \
    cd netcdf && \
    CPPFLAGS="-I${TPLS_ROOT}/include" LDFLAGS="-L${TPLS_ROOT}/lib" \
    ./configure --prefix=${TPLS_ROOT} --enable-shared --disable-static \
                --enable-parallel4 --enable-pnetcdf \
                --disable-dap --disable-byterange --disable-nczarr --disable-libxml2 \
                --disable-testsets && \
    make -j8 install && \
    cd ${TPLS_ROOT}/src && rm -rf netcdf netcdf.tar.gz && \
    nc-config --all

ENV CUDA_HOME=/usr/local/cuda

# TRILINOS
ARG TRILINOS_COMMIT=origin/develop

RUN cd $SOFTWARE_ROOT && \
    mkdir trilinos && cd trilinos && \
    git clone https://github.com/trilinos/trilinos source && \
    cd source && \
    git checkout $TRILINOS_COMMIT

# Same as Perlmutter's nvcc_wrapper_a100: Kokkos' nvcc_wrapper (default arch sm_80)
# with the MPI C++ compiler as default host compiler (CC on Perlmutter, mpicxx here)
ENV NVCC_WRAPPER=${SOFTWARE_ROOT}/bin/nvcc_wrapper
RUN mkdir -p ${SOFTWARE_ROOT}/bin && \
    cp ${SOFTWARE_ROOT}/trilinos/source/packages/kokkos/bin/nvcc_wrapper ${NVCC_WRAPPER} && \
    sed -i 's/NVCC_WRAPPER_DEFAULT_COMPILER:-"g++"/NVCC_WRAPPER_DEFAULT_COMPILER:-"mpicxx"/' ${NVCC_WRAPPER} && \
    grep -q 'NVCC_WRAPPER_DEFAULT_COMPILER:-"mpicxx"' ${NVCC_WRAPPER} && \
    grep -q '^default_arch="sm_80"' ${NVCC_WRAPPER} && \
    chmod 775 ${NVCC_WRAPPER}

ADD --chown=runner:fanssie --chmod=775 do-cmake-trilinos.sh ${SOFTWARE_ROOT}/trilinos/do-cmake-trilinos.sh

# Configure, build, install, and clean up in a single layer: a CUDA build
# tree is large, and we don't want it to be part of the image.
ARG NUM_JOBS=4
RUN mkdir $SOFTWARE_ROOT/trilinos/build && \
    cd $SOFTWARE_ROOT/trilinos/build && \
    ../do-cmake-trilinos.sh && \
    make -j${NUM_JOBS} install && \
    cd $SOFTWARE_ROOT/trilinos && \
    rm -rf source build

ENV TRILINOS_ROOT=${SOFTWARE_ROOT}/trilinos/install

CMD [ "/bin/bash"]
