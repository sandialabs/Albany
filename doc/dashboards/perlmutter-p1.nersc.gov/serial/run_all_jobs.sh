#!/bin/csh

ulimit -c 0

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-serial

source ${BASE_DIR}/pm_cpu_gnu_modules.sh >& ${BASE_DIR}/modules_albany.out

bash ${BASE_DIR}/nightly_cron_script_trilinos_pm_cpu.sh
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${BASE_DIR}/build/TrilinosSerialInstallGcc/lib"

bash ${BASE_DIR}/nightly_cron_script_albany_pm_cpu.sh
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${BASE_DIR}/build/AlbanySerialInstallGcc/lib64"

bash ${BASE_DIR}/nightly_cron_script_mali_pm_cpu.sh

mkdir ${BASE_DIR}/repos/compass_tests
cp ${BASE_DIR}/CTestTestfile.cmake ${BASE_DIR}/repos/compass_tests
cp ${BASE_DIR}/test_and_log.sh ${BASE_DIR}/repos/compass_tests
bash ${BASE_DIR}/nightly_cron_script_compass_pm_cpu.sh