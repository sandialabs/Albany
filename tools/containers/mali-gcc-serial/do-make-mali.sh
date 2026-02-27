#!/bin/bash
make clean

source ${SOFTWARE_ROOT}/albany/install/export_albany.in

export LD_LIBRARY_PATH=${PNETCDF_ROOT}/lib:${LD_LIBRARY_PATH}

ARCH=$(uname -m)

if [ "${ARCH}" = "x86_64" ]; then
  make -j 4 gfortran \
    ALBANY=true \
    USE_PIO2=true \
    CORE=landice \
    SLM=true \
    PIO=${SCORPIO_ROOT} \
    NETCDF=${NETCDF_F_ROOT} \
    MPAS_EXTERNAL_LIBS="${ALBANY_LINK_LIBS} -Wl,-R${ALBANY_ROOT}/lib64 -Wl,-R${TRILINOS_ROOT}/lib64" \
    DEBUG=false \
    EXE_NAME=landice_model;
else
  GFORTRAN_GTE_10=$(expr "$(gfortran -dumpversion | cut -f1 -d.)" \>= 10)
  if [ "${GFORTRAN_GTE_10}" = "1" ]; then
    EXTRA_FFLAGS="-fallow-argument-mismatch"
  else
    EXTRA_FFLAGS=""
  fi

  make -j 4 all \
    FC_PARALLEL=mpif90 \
    CC_PARALLEL=mpicc \
    CXX_PARALLEL=mpicxx \
    FC_SERIAL=gfortran \
    CC_SERIAL=gcc \
    CXX_SERIAL=g++ \
    FFLAGS_FPIEEE= \
    FFLAGS_PROMOTION="-fdefault-real-8 -fdefault-double-8" \
    FFLAGS_OPT="-O3 -ffree-line-length-none -fconvert=big-endian -ffree-form -ffpe-summary=none ${EXTRA_FFLAGS}" \
    CFLAGS_OPT="-O3" \
    CXXFLAGS_OPT="-O3" \
    LDFLAGS_OPT="-O3" \
    FFLAGS_DEBUG="-g -ffree-line-length-none -fconvert=big-endian -ffree-form -fbounds-check -fbacktrace -ffpe-trap=invalid,zero,overflow -ffpe-summary=none ${EXTRA_FFLAGS}" \
    CFLAGS_DEBUG="-g" \
    CXXFLAGS_DEBUG="-O3" \
    LDFLAGS_DEBUG="-g" \
    FFLAGS_OMP="-fopenmp" \
    CFLAGS_OMP="-fopenmp" \
    PICFLAG="-fPIC" \
    BUILD_TARGET=gfortran \
    CPPFLAGS="-D_MPI" \
    ALBANY=true \
    USE_PIO2=true \
    CORE=landice \
    SLM=true \
    PIO=${SCORPIO_ROOT} \
    NETCDF=${NETCDF_F_ROOT} \
    MPAS_EXTERNAL_LIBS="${ALBANY_LINK_LIBS} -Wl,-R${ALBANY_ROOT}/lib64 -Wl,-R${TRILINOS_ROOT}/lib64" \
    DEBUG=false \
    EXE_NAME=landice_model;
fi