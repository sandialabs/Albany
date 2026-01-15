#!/bin/bash
make clean

source ${SOFTWARE_ROOT}/albany/install/export_albany.in

export LD_LIBRARY_PATH=${PNETCDF_ROOT}/lib:${LD_LIBRARY_PATH}

make -j gfortran \
  ALBANY=true \
  USE_PIO2=true \
  CORE=landice \
  PIO=${SCORPIO_ROOT} \
  NETCDF=${NETCDF_F_ROOT} \
  MPAS_EXTERNAL_LIBS="${ALBANY_LINK_LIBS} -Wl,-R${ALBANY_ROOT}/lib64 -Wl,-R${TRILINOS_ROOT}/lib64" \
  DEBUG=false \
  EXE_NAME=landice_model;
