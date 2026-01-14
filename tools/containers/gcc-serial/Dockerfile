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

ENV SOFTWARE_ROOT=/software
RUN mkdir $SOFTWARE_ROOT

# TRILINOS
RUN cd $SOFTWARE_ROOT && \
    mkdir trilinos && cd trilinos && \
    git clone --filter=blob:none https://github.com/trilinos/trilinos source

RUN mkdir $SOFTWARE_ROOT/trilinos/build

ADD do-cmake-trilinos.sh ${SOFTWARE_ROOT}/trilinos/build/do-cmake-trilinos.sh

RUN cd $SOFTWARE_ROOT/trilinos/build && \
    chmod u+x do-cmake-trilinos.sh && ./do-cmake-trilinos.sh

RUN cd $SOFTWARE_ROOT/trilinos/build && \
    make -j16 install

ENV TRILINOS_ROOT=${SOFTWARE_ROOT}/trilinos/install

RUN cd $SOFTWARE_ROOT/trilinos && \
    mv build/do-cmake-trilinos.sh ./    && \
    rm -rf source build

RUN useradd -ms /bin/bash runner

# Install sudo, set password for 'runner', and configure sudoers. Just in case we want to install
# some packages while we're running the container
RUN apt-get update && \
    apt-get install -y sudo && \
    echo "runner:fannsie" | chpasswd && \
    echo "runner ALL=(ALL) ALL" >> /etc/sudoers && \
    apt-get clean

# Switch to user, and set some useful bindings
USER runner

RUN echo 'bind "\"\\e[A\": history-search-backward"' >> /home/runner/.bashrc
RUN echo 'bind "\"\\e[B\": history-search-forward"' >> /home/runner/.bashrc

CMD [ "/bin/bash"]
