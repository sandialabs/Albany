FROM ghcr.io/sandialabs/albany:gcc-12-serial

# Albany
RUN cd $SOFTWARE_ROOT && \
    mkdir albany && cd albany && \
    git clone --filter=blob:none https://github.com/sandialabs/albany source

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
    apt-get install -y libnetcdff-dev
USER runner

# Set up folders with symlinks for netcdf-f and pnetcdf, as scorpio's clunky cmake logic cannot
# find them (since include and lib are not in the same root folder)
RUN mkdir $SOFTWARE_ROOT/netcdf-f && \
    cd $SOFTWARE_ROOT/netcdf-f &&    \
    ln -s /usr/include &&            \
    ln -s /usr/lib/x86_64-linux-gnu/ lib
ENV NETCDF_F_ROOT=${SOFTWARE_ROOT}/netcdf-f
RUN mkdir $SOFTWARE_ROOT/pnetcdf && \
    cd $SOFTWARE_ROOT/pnetcdf && \
    ln -s /usr/include && \
    ln -s /usr/lib/x86_64-linux-gnu/ lib
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
RUN cd $SOFTWARE_ROOT && \
    mkdir mali && cd mali && \
    git clone --filter=blob:none https://github.com/mali-dev/e3sm source && \
    git config --global url."https://github.com/".insteadOf git@github.com: && \
    cd source/components/mpas-albany-landice/src && git submodule update --init SeaLevelModel

ADD --chown=runner:fanssie --chmod=775 do-make-mali.sh $SOFTWARE_ROOT/mali/source/components/mpas-albany-landice

RUN cd $SOFTWARE_ROOT/mali/source/components/mpas-albany-landice && \
    cat do-make-mali.sh && \
    ./do-make-mali.sh

# Setup env vars
ENV LD_LIBRARY_PATH=${TRILINOS_ROOT}/lib

# set up conda
USER runner
RUN wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O /home/runner/Miniforge3.sh && \
    bash /home/runner/Miniforge3.sh -b -p /home/runner/miniforge && \
    rm -f /home/runner/Miniforge3.sh

ENV PATH="/home/runner/miniforge/bin:${PATH}"

# set up compass
# Note: the -fno-lto flags were needed to fix an internal compiler.
# This may be specific to my machine architecture and the compiler version being used here and should not
# generally be required.  But I don't think they are going to hurt either, as we don't expect to use
# jigsaw much in this container setup.
RUN conda create -n compass -c e3sm/label/compass -y -c conda-forge python=3.13 "compass=1.8.0=mpi_mpich*"

# build jigsaw for compass
WORKDIR /tmp
RUN /home/runner/miniforge/bin/conda run -n compass bash -lc '\
    export CFLAGS="-O2 -fno-lto" && \
    export CXXFLAGS="-O2 -fno-lto" && \
    export LDFLAGS="-fno-lto" && \
    build_jigsaw --clone --subdir jigsaw-python'

# create compass load script
WORKDIR /home/runner
RUN /bin/bash -lc 'source /home/runner/miniforge/etc/profile.d/conda.sh && \
    conda activate compass && \
    create_compass_load_script'

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
