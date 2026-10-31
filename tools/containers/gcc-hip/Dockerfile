# Build-only gcc + HIP (ROCm) container, loosely based on the Frontier HIP build
# (see doc/dashboards/frontier.lb.ccs.ornl.gov). It does not reproduce the
# Frontier environment exactly: to keep the image small, it reuses the
# gcc-serial dependency stack (ubuntu packages, OpenMPI) and only installs
# the parts of ROCm 6.2.4 (Frontier's version) needed to compile Trilinos/Albany.
# As on Frontier, hipcc is the C++ compiler, and MPI is added with explicit
# include/link flags (here taken from OpenMPI's mpicxx --showme), while C and
# Fortran use the gcc-based MPI wrappers (mpicc/mpifort).
# There are no GPUs on github runners, so this image can only be used to build.
FROM docker.io/ubuntu:24.04

LABEL maintainer="jwatkin@sandia.gov"

ENV DEBIAN_FRONTEND=noninteractive

# Install basic utils
RUN apt-get update &&   \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl            \
        git             \
        cmake           \
        make            \
        locate          \
        vim             \
        wget         && \
    rm -rf /var/lib/apt/lists/*

# Install compilers and python
RUN apt-get update &&   \
    apt-get install -y --no-install-recommends \
        gcc             \
        g++             \
        gfortran        \
        libc6-dev       \
        python3      && \
    rm -rf /var/lib/apt/lists/*

# Install MPI
RUN apt-get update &&   \
    apt-get install -y --no-install-recommends \
        libopenmpi-dev  \
        openmpi-bin  && \
    rm -rf /var/lib/apt/lists/*

# Install blas, yaml-cpp, boost
RUN apt-get update &&   \
    apt-get install -y --no-install-recommends \
        libyaml-cpp-dev \
        libboost-dev    \
        libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

# Install hdf5, netcdf, pnetcdf, metis, parmetis
RUN apt-get update &&       \
    apt-get install -y --no-install-recommends \
        libhdf5-openmpi-dev \
        libnetcdf-mpi-dev   \
        netcdf-bin          \
        libpnetcdf-dev      \
        libmetis-dev        \
        libparmetis-dev  && \
    rm -rf /var/lib/apt/lists/*

# Install a minimal ROCm (compiler, HIP runtime, and the math libs Trilinos looks for).
# The math libs ship kernels for many GPU archs: to save disk, we remove the
# rocBLAS (Tensile) kernel files of every arch other than gfx90a (MI250X).
# The pin makes apt prefer AMD's packages over ubuntu's own (older) ROCm packages.
ARG ROCM_VERSION=6.2.4
RUN mkdir -p /etc/apt/keyrings && \
    wget -q -O /etc/apt/keyrings/rocm.asc https://repo.radeon.com/rocm/rocm.gpg.key && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.asc] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
        > /etc/apt/sources.list.d/rocm.list && \
    printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
        > /etc/apt/preferences.d/rocm-pin-600 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        hipcc            \
        hip-dev          \
        rocm-device-libs \
        rocblas-dev      \
        rocsparse-dev    \
        rocsolver-dev    \
        rocprim-dev      \
        rocthrust-dev && \
    find /opt/rocm-${ROCM_VERSION}/lib/rocblas/library -name '*gfx*' ! -name '*gfx90a*' -delete && \
    rm -rf /var/lib/apt/lists/*

ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=${ROCM_PATH}
ENV PATH=${ROCM_PATH}/bin:${PATH}

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
    apt-get install -y --no-install-recommends sudo && \
    echo "runner:fanssie" | chpasswd && \
    echo "runner ALL=(ALL) ALL" >> /etc/sudoers && \
    rm -rf /var/lib/apt/lists/*

ENV SOFTWARE_ROOT=/software
RUN mkdir $SOFTWARE_ROOT && \
    chown :fanssie $SOFTWARE_ROOT && \
    chmod 2770 $SOFTWARE_ROOT

# Switch to user from now on
USER runner

# TRILINOS
ARG TRILINOS_COMMIT=origin/develop

RUN cd $SOFTWARE_ROOT && \
    mkdir trilinos && cd trilinos && \
    git clone --filter=blob:none https://github.com/trilinos/trilinos source && \
    cd source && \
    git checkout $TRILINOS_COMMIT

ADD --chown=runner:fanssie --chmod=775 do-cmake-trilinos.sh ${SOFTWARE_ROOT}/trilinos/do-cmake-trilinos.sh

# Configure, build, install, and clean up in a single layer: a HIP build
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
