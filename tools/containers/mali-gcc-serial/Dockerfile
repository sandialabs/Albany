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

CMD [ "/bin/bash" ]
