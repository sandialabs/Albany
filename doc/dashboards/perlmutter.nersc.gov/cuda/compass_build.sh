#!/bin/bash

BASE_DIR=/pscratch/sd/j/jwatkins/nightlyCDashPerlmutterCuda

cd ${BASE_DIR}/repos/compass

# compass now deploys with ./deploy.py (pixi + mache.deploy); pixi must be on
# PATH or at ~/.pixi/bin/pixi
# pixi/rattler lock their cache; /global/homes does not support those locks
# from scron (os error 524), so keep the cache on pscratch
export PIXI_CACHE_DIR=${BASE_DIR}/pixi_cache
export RATTLER_CACHE_DIR=${PIXI_CACHE_DIR}
git submodule update --init jigsaw-python
./deploy.py --machine pm-gpu --compiler gnugpu --mpi mpich --no-spack --pixi-path ${BASE_DIR}/build/compass --recreate 2>&1 | tee deploy_compass_output.txt
LOAD_SCRIPT=`ls -t ${BASE_DIR}/repos/compass/load_compass_pm-gpu*.sh 2>/dev/null | head -1`
if [ -z "${LOAD_SCRIPT}" ]; then
  echo "Error: compass deploy did not produce a load_compass_pm-gpu*.sh script"
  exit 1
fi
source ${LOAD_SCRIPT}

cd ${BASE_DIR}
compass suite -c landice -t full_integration -w ${BASE_DIR}/repos/compass_tests -s -f ${BASE_DIR}/compass_pm-gpu_config.cfg -p ${BASE_DIR}/repos/E3SM/components/mpas-albany-landice -m pm-gpu
