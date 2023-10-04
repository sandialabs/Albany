BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-serial
SCORPIO_DIR=/global/common/software/fanssie/scorpio-gnu

cd ${BASE_DIR}/repos/E3SM/components/mpas-albany-landice

source /global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial/builds/AlbanyInstall/export_albany.in
echo ${ALBANY_LINK_LIBS}

echo ${LD_LIBRARY_PATH}

make clean
make -j 12 gnu-cray \
  ALBANY=true \
  USE_PIO2=true \
  CORE=landice \
  PIO=${SCORPIO_DIR} \
  NETCDF=${NETCDF_DIR} \
  MPAS_EXTERNAL_LIBS="${ALBANY_LINK_LIBS}" \
  DEBUG=false \
  EXE_NAME=landice_model;

mkdir /global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial/builds/mali
cp landice_model /global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial/builds/mali/landice_model