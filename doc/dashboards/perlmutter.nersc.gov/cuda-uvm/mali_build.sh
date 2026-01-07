BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-cuda-uvm
DEPLOY_DIR=/global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/cuda-uvm

# Build SCORPIO
bash ${BASE_DIR}/do-cmake-scorpio-gcc-release

SCORPIO_DIR=${DEPLOY_DIR}/builds/scorpio-gnu
ALBANY_INSTALL_DIR=${DEPLOY_DIR}/builds/AlbanyInstallSfad12
TRILINOS_INSTALL_DIR=${DEPLOY_DIR}/builds/TrilinosInstall

cd ${BASE_DIR}/repos/E3SM/components/mpas-albany-landice

source ${DEPLOY_DIR}/builds/AlbanyInstall/export_albany.in
echo ${ALBANY_LINK_LIBS}

echo ${LD_LIBRARY_PATH}

make clean
make -j 12 gnu-cray \
  ALBANY=true \
  USE_PIO2=true \
  CORE=landice \
  PIO=${SCORPIO_DIR} \
  NETCDF=${NETCDF_DIR} \
  PNETCDF=${PNETCDF_DIR} \
  MPAS_EXTERNAL_LIBS="${ALBANY_LINK_LIBS} -Wl,-R${ALBANY_INSTALL_DIR}/lib64 -Wl,-R${TRILINOS_INSTALL_DIR}/lib64" \
  DEBUG=false \
  EXE_NAME=landice_model;

mkdir ${DEPLOY_DIR}/builds/mali
cp landice_model ${DEPLOY_DIR}/builds/mali/landice_model
cp -r default_inputs ${DEPLOY_DIR}/builds/mali/
