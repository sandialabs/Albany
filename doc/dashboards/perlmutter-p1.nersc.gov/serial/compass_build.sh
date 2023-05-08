#!/bin/csh

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-serial

cd ${BASE_DIR}/repos/compass

python ./conda/configure_compass_env.py --conda ${BASE_DIR}/build/compass --mpi mpich --with_albany -m pm-cpu
source load_dev_compass_1.2.0-alpha.5_pm-cpu_gnu_mpich_albany.sh 

cd ${BASE_DIR}
compass suite -c landice -t full_integration -w ${BASE_DIR}/repos/compass_tests -s -f ${BASE_DIR}/compass_pm-cpu_config.cfg -p ${BASE_DIR}/repos/E3SM/components/mpas-albany-landice
