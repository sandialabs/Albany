BASE_DIR=/lustre/orion/cli193/scratch/mcarlson/testingCDashFrontier-rocm
DEPLOY_DIR=/lustre/orion/cli193/proj-shared/automated_testing/rocm
ALBANY_INSTALL_DIR=${DEPLOY_DIR}/builds/AlbanyInstallSfad12
TRILINOS_INSTALL_DIR=${DEPLOY_DIR}/builds/TrilinosInstall

SCORPIO_DIR=/lustre/orion/cli193/proj-shared/automated_testing/scorpio-gnu

cd ${BASE_DIR}/repos/E3SM/components/mpas-albany-landice

source ${ALBANY_INSTALL_DIR}/export_albany.in
export ALBANY_LINK_LIBS=${ALBANY_LINK_LIBS//"--hip-link "}
export ALBANY_LINK_LIBS=${ALBANY_LINK_LIBS//"--offload-arch=gfx90a "}
export ALBANY_LINK_LIBS=${ALBANY_LINK_LIBS//"-lclang_rt "}
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

