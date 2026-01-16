FROM docker.io/ubuntu:latest

LABEL maintainer="lbertag@sandia.gov"

ENV DEBIAN_FRONTEND=noninteractive

# Install basic utils
RUN apt-get update &&   \
    apt-get install -y  \
        curl            \
        git             \
        cmake           \
        locate          \
        vim             \
        wget

# Install compilers and python
RUN apt-get update &&   \
    apt-get install -y  \
        build-essential \
        gcc             \
        g++             \
        gfortran        \
        python3

# Install MPI
RUN apt-get update &&   \
    apt-get install -y  \
        libopenmpi-dev

# Install blas, yaml-cpp, boost
RUN apt-get update &&   \
    apt-get install -y  \
        libyaml-cpp-dev \
        libboost-dev    \
        libopenblas-dev

# Install hdf5, netcdf, pnetcdf, metis, parmetis
RUN apt-get update &&       \
    apt-get install -y      \
        libhdf5-openmpi-dev \
        libnetcdf-mpi-dev   \
        libpnetcdf-dev      \
        libmetis-dev        \
        libparmetis-dev

ENV NETCDF_C_ROOT=/usr/lib/x86_64-linux-gnu/netcdf/mpi

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
# This helkps if we realize we need an extra pkg while running a container
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

# TRILINOS
RUN cd $SOFTWARE_ROOT && \
    mkdir trilinos && cd trilinos && \
    git clone --filter=blob:none https://github.com/trilinos/trilinos source

RUN mkdir $SOFTWARE_ROOT/trilinos/build

ADD --chown=runner:fanssie --chmod=775 do-cmake-trilinos.sh ${SOFTWARE_ROOT}/trilinos/build/do-cmake-trilinos.sh

RUN cd $SOFTWARE_ROOT/trilinos/build && \
    ./do-cmake-trilinos.sh

RUN cd $SOFTWARE_ROOT/trilinos/build && \
    make -j16 install

ENV TRILINOS_ROOT=${SOFTWARE_ROOT}/trilinos/install

RUN cd $SOFTWARE_ROOT/trilinos && \
    mv build/do-cmake-trilinos.sh ./    && \
    rm -rf source build

CMD [ "/bin/bash"]
