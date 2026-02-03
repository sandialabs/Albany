#!/bin/csh

BASE_DIR=/pscratch/sd/j/jwatkins/nightlyCDashPerlmutterCuda

cd ${BASE_DIR}/repos/compass

./conda/configure_compass_env.py --conda ${BASE_DIR}/build/compass --env_only -m pm-gpu |& tee configure_compass_env_output.txt
LOAD_DEV_FILENAME=`grep 'load_dev' configure_compass_env_output.txt | sed 's/ //g'`
source $LOAD_DEV_FILENAME

cd ${BASE_DIR}
compass suite -c landice -t full_integration -w ${BASE_DIR}/repos/compass_tests -s -f ${BASE_DIR}/compass_pm-gpu_config.cfg -p ${BASE_DIR}/repos/E3SM/components/mpas-albany-landice -m pm-gpu
