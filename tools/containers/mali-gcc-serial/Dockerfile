# syntax=docker/dockerfile:1.4
ARG BASE_IMAGE=ghcr.io/sandialabs/albany:gcc-12-serial
FROM ${BASE_IMAGE}
# Global TLS switch for restrictive networks.
# WARNING: when true, TLS cert verification is disabled for git/wget/conda in this Docker build.
ARG ALLOW_INSECURE_CERTS=false
ENV ALLOW_INSECURE_CERTS=${ALLOW_INSECURE_CERTS}
ENV GIT_SSL_NO_VERIFY=${ALLOW_INSECURE_CERTS}

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
    git submodule sync -- components/mpas-albany-landice/src/SeaLevelModel && \
    git submodule update --init --recursive components/mpas-albany-landice/src/SeaLevelModel && \
    test -e components/mpas-albany-landice/src/SeaLevelModel

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
    if [ "${ALLOW_INSECURE_CERTS}" = "true" ]; then \
        wget --no-check-certificate --tries=5 --retry-connrefused --waitretry=2 \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${MINIFORGE_ARCH}.sh" -O /home/runner/Miniforge3.sh; \
    else \
        wget --tries=5 --retry-connrefused --waitretry=2 \
        --ca-certificate=/etc/ssl/certs/ca-certificates.crt \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${MINIFORGE_ARCH}.sh" -O /home/runner/Miniforge3.sh; \
    fi

RUN bash /home/runner/Miniforge3.sh -b -p /home/runner/miniforge && \
    rm -f /home/runner/Miniforge3.sh

RUN if [ "${ALLOW_INSECURE_CERTS}" = "true" ]; then \
        /home/runner/miniforge/bin/conda config --system --set ssl_verify false; \
    else \
        /home/runner/miniforge/bin/conda config --system --set ssl_verify /etc/ssl/certs/ca-certificates.crt; \
    fi

ENV PATH="/home/runner/miniforge/bin:${PATH}"

# set up compass (developer installation, env only)
ARG COMPASS_COMMIT=origin/main
WORKDIR /home/runner
RUN git clone --filter=blob:none https://github.com/MPAS-Dev/compass /home/runner/compass && \
    cd /home/runner/compass && \
    git checkout ${COMPASS_COMMIT}

# Create/update only the compass conda environment and load script from the repo.
# This performs a development installation of compass from /home/runner/compass.
RUN if [ "$(uname -m)" = "x86_64" ]; then \
            cd /home/runner/compass && \
            if [ "${ALLOW_INSECURE_CERTS}" = "true" ]; then CONDA_SSL_VERIFY=false; else CONDA_SSL_VERIFY=/etc/ssl/certs/ca-certificates.crt; fi && \
            /bin/bash -lc 'export CFLAGS="-O2 -fno-lto" && \
            export CXXFLAGS="-O2 -fno-lto" && \
            export LDFLAGS="-fno-lto" && \
            export CONDA_SSL_VERIFY="'"${CONDA_SSL_VERIFY}"'" && \
            source /home/runner/miniforge/etc/profile.d/conda.sh && \
            if [ "'"${ALLOW_INSECURE_CERTS}"'" = "true" ]; then \
                unset REQUESTS_CA_BUNDLE SSL_CERT_FILE CURL_CA_BUNDLE && \
                export SSL_NO_VERIFY=1 && \
                conda config --set ssl_verify false; \
            else \
                conda config --set ssl_verify /etc/ssl/certs/ca-certificates.crt; \
            fi && \
            ./conda/configure_compass_env.py --conda /home/runner/miniforge --env_only'; \
        else \
            echo "Skipping Compass developer env setup on $(uname -m): required deps are not available for linux-aarch64."; \
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
