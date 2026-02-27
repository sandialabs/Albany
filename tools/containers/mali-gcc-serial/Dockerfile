ARG BASE_IMAGE=ghcr.io/sandialabs/albany:gcc-12-serial
FROM ${BASE_IMAGE}

# Albany
ARG ALBANY_COMMIT=origin/HEAD
RUN cd $SOFTWARE_ROOT && \
    mkdir albany && cd albany && \
    git clone --filter=blob:none https://github.com/sandialabs/albany source && \
    cd source && \
    git checkout ${ALBANY_COMMIT}

RUN mkdir $SOFTWARE_ROOT/albany/build

ADD --chown=runner:fanssie --chmod=775 do-cmake-albany.sh ${SOFTWARE_ROOT}/albany/build/do-cmake-albany.sh

RUN cd $SOFTWARE_ROOT/albany/build && \
    ./do-cmake-albany.sh

RUN cd $SOFTWARE_ROOT/albany/build && \
    make -j8 install

ENV ALBANY_ROOT=${SOFTWARE_ROOT}/albany/install

RUN cd $SOFTWARE_ROOT/albany && \
    mv build/do-cmake-albany.sh ./  && \
    rm -rf source build

# HACK 1: for some reason, mali wants to use libmpi_cxx, but I get a link error.
#         So add it to the list of albany libs, and cross your fingers...
RUN cd $SOFTWARE_ROOT/albany/install && \
    sed -i 's/-ldl/-ldl -lmpi_cxx/g' export_albany.in

# Switch back to root to install netcdff
USER root
RUN apt-get update && \
    apt-get install -y ca-certificates netcdf-bin libnetcdff-dev libpnetcdf-dev && \
    update-ca-certificates && \
    update-ca-certificates --fresh 
USER runner

# Set up folders with symlinks for netcdf-f and pnetcdf, as scorpio's clunky cmake logic cannot
# find them (since include and lib are not in the same root folder)
RUN NETCDF_C_INCLUDE_DIR="$(nc-config --includedir 2>/dev/null || echo /usr/include)" && \
    NETCDF_C_LIB_DIR="$(nc-config --libdir 2>/dev/null || echo /usr/lib/$(gcc -print-multiarch))" && \
    mkdir $SOFTWARE_ROOT/netcdf-c && \
    cd $SOFTWARE_ROOT/netcdf-c &&    \
    ln -s "${NETCDF_C_INCLUDE_DIR}" include && \
    ln -s "${NETCDF_C_LIB_DIR}" lib
ENV NETCDF_C_ROOT=${SOFTWARE_ROOT}/netcdf-c
RUN NETCDF_F_INCLUDE_DIR="$(nf-config --includedir 2>/dev/null || echo /usr/include)" && \
    NETCDF_F_LIB_DIR="$(nf-config --flibs 2>/dev/null | sed -n 's/.*-L\([^ ]*\).*/\1/p')" && \
    if [ -z "${NETCDF_F_LIB_DIR}" ]; then NETCDF_F_LIB_DIR="/usr/lib/$(gcc -print-multiarch)"; fi && \
    mkdir $SOFTWARE_ROOT/netcdf-f && \
    cd $SOFTWARE_ROOT/netcdf-f &&    \
    ln -s "${NETCDF_F_INCLUDE_DIR}" include && \
    ln -s "${NETCDF_F_LIB_DIR}" lib
ENV NETCDF_F_ROOT=${SOFTWARE_ROOT}/netcdf-f
RUN PNETCDF_INCLUDE_DIR="$(pnetcdf-config --includedir 2>/dev/null || echo /usr/include)" && \
    PNETCDF_LIB_DIR="$(pnetcdf-config --libdir 2>/dev/null || echo /usr/lib/$(gcc -print-multiarch))" && \
    mkdir $SOFTWARE_ROOT/pnetcdf && \
    cd $SOFTWARE_ROOT/pnetcdf && \
    ln -s "${PNETCDF_INCLUDE_DIR}" include && \
    ln -s "${PNETCDF_LIB_DIR}" lib
ENV PNETCDF_ROOT=${SOFTWARE_ROOT}/pnetcdf

# SCORPIO
RUN cd $SOFTWARE_ROOT && \
    mkdir scorpio && cd scorpio && \
    git clone --filter=blob:none https://github.com/e3sm-project/scorpio source

RUN mkdir $SOFTWARE_ROOT/scorpio/build

ADD --chown=runner:fanssie --chmod=775 do-cmake-scorpio.sh ${SOFTWARE_ROOT}/scorpio/build/do-cmake-scorpio.sh

RUN cd $SOFTWARE_ROOT/scorpio/build && \
    ./do-cmake-scorpio.sh

RUN cd $SOFTWARE_ROOT/scorpio/build && \
    make -j8 install

ENV SCORPIO_ROOT=${SOFTWARE_ROOT}/scorpio/install

RUN cd $SOFTWARE_ROOT/scorpio && \
    mv build/do-cmake-scorpio.sh ./  && \
    rm -rf source build

# MALI
ARG MALI_ORG=mali-dev
ARG MALI_COMMIT=origin/develop
RUN cd $SOFTWARE_ROOT && \
    mkdir mali && cd mali && \
    git clone --filter=blob:none https://github.com/${MALI_ORG}/e3sm source && \
    cd source && \
    git checkout ${MALI_COMMIT} && \
    git config --global url."https://github.com/".insteadOf git@github.com: && \
    cd components/mpas-albany-landice/src && git submodule update --init SeaLevelModel

ADD --chown=runner:fanssie --chmod=775 do-make-mali.sh $SOFTWARE_ROOT/mali/source/components/mpas-albany-landice

RUN cd $SOFTWARE_ROOT/mali/source/components/mpas-albany-landice && \
    cat do-make-mali.sh && \
    ./do-make-mali.sh

# Setup env vars
ENV LD_LIBRARY_PATH=${TRILINOS_ROOT}/lib

# set up conda
USER runner
ARG TARGETARCH
RUN if [ "${TARGETARCH}" = "amd64" ]; then MINIFORGE_ARCH="x86_64"; \
    elif [ "${TARGETARCH}" = "arm64" ]; then MINIFORGE_ARCH="aarch64"; \
    else MINIFORGE_ARCH="${TARGETARCH}"; fi && \
    echo "TARGETARCH=${TARGETARCH}" && \
    echo "uname -m=$(uname -m)" && \
    echo "MINIFORGE_ARCH=${MINIFORGE_ARCH}" && \
    echo "MINIFORGE_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${MINIFORGE_ARCH}.sh" && \
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${MINIFORGE_ARCH}.sh" -O /home/runner/Miniforge3.sh 

RUN bash /home/runner/Miniforge3.sh -b -p /home/runner/miniforge && \
    rm -f /home/runner/Miniforge3.sh

ENV PATH="/home/runner/miniforge/bin:${PATH}"

# set up compass
# NOTE: Compass is currently skipped on ARM (aarch64).
# The e3sm/label/compass + conda-forge channels do not currently provide
# all required dependencies on linux-aarch64 (notably moab and otps), so
# `conda create` for compass fails on ARM at solve time.
# Note: the -fno-lto flags were needed to fix an internal compiler.
# This may be specific to my machine architecture and the compiler version being used here and should not
# generally be required.  But I don't think they are going to hurt either, as we don't expect to use
# jigsaw much in this container setup.
RUN if [ "$(uname -m)" = "x86_64" ]; then \
            conda create -n compass -c e3sm/label/compass -y -c conda-forge python=3.13 "compass=1.8.0=mpi_mpich*"; \
        else \
            echo "Skipping Compass conda env on $(uname -m): required deps (moab, otps) are not available for linux-aarch64."; \
        fi

# build jigsaw for compass
WORKDIR /tmp
RUN if [ "$(uname -m)" = "x86_64" ]; then \
            /home/runner/miniforge/bin/conda run -n compass bash -lc '\
            export CFLAGS="-O2 -fno-lto" && \
            export CXXFLAGS="-O2 -fno-lto" && \
            export LDFLAGS="-fno-lto" && \
            build_jigsaw --clone --subdir jigsaw-python'; \
        else \
            echo "Skipping JIGSAW build because Compass env is not available on $(uname -m)."; \
        fi

# create compass load script
WORKDIR /home/runner
RUN if [ "$(uname -m)" = "x86_64" ]; then \
            /bin/bash -lc 'source /home/runner/miniforge/etc/profile.d/conda.sh && \
            conda activate compass && \
            create_compass_load_script'; \
        else \
            echo "Skipping compass load script on $(uname -m): Compass env not installed."; \
        fi

# add the compass cfg file specific to the container
ADD --chown=runner:fanssie --chmod=775 compass-container.cfg /home/runner

# update environment for user
RUN cat >> /home/runner/.bashrc <<'EOF'
export CLICOLOR=1
export PS1="\[\e[32;1m\]\t \u@\h \[\e[0m\]\[\e[33;1m\]\w\[\e[0m\]\[\e[32;1m\]\$\[\e[0m\] "
bind "\"\e[A\": history-search-backward"
bind "\"\e[B\": history-search-forward"
EOF

# add entrypoint file to load customize env for the user
COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
