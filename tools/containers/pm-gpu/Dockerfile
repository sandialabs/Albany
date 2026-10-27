# Build-only CUDA container, loosely modeled on the Perlmutter GPU (pm-gpu) build
# (see doc/dashboards/perlmutter.nersc.gov/cuda). It does not reproduce the
# Perlmutter environment exactly: to keep the image small, it reuses the
# gcc-serial dependency stack (ubuntu packages, OpenMPI) and only installs
# the parts of CUDA 12.9 needed to compile Trilinos/Albany.
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

# Install a minimal CUDA toolkit (compiler, runtime, and the math libs Trilinos looks for)
ARG CUDA_VERSION=12-9
RUN cd /tmp && \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-nvcc-${CUDA_VERSION}          \
        cuda-cudart-dev-${CUDA_VERSION}    \
        cuda-cccl-${CUDA_VERSION}          \
        cuda-nvtx-${CUDA_VERSION}          \
        cuda-nvml-dev-${CUDA_VERSION}      \
        cuda-profiler-api-${CUDA_VERSION}  \
        libcublas-dev-${CUDA_VERSION}      \
        libcufft-dev-${CUDA_VERSION}       \
        libcusolver-dev-${CUDA_VERSION}    \
        libcusparse-dev-${CUDA_VERSION} && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}

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

# Like Perlmutter's nvcc_wrapper_a100: Kokkos' nvcc_wrapper (default arch sm_80)
# using the MPI C++ compiler as default host compiler
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
