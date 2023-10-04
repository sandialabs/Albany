#!/bin/csh

ulimit -c 0

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-serial

source ${BASE_DIR}/pm_cpu_gnu_modules.sh >& ${BASE_DIR}/modules_albany.out

bash ${BASE_DIR}/nightly_cron_script_trilinos_pm_cpu.sh
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial/builds/TrilinosInstall/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial/builds/TrilinosInstall/lib64"

bash ${BASE_DIR}/nightly_cron_script_albany_pm_cpu.sh
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial/builds/AlbanyInstall/lib64"

bash ${BASE_DIR}/nightly_cron_script_mali_pm_cpu.sh

mkdir ${BASE_DIR}/repos/compass_tests
cp ${BASE_DIR}/CTestTestfile.cmake ${BASE_DIR}/repos/compass_tests
cp ${BASE_DIR}/test_and_log.sh ${BASE_DIR}/repos/compass_tests
bash ${BASE_DIR}/nightly_cron_script_compass_pm_cpu.sh

cp biweekly_log_pm_cpu_Albany.txt /global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial/logs/biweekly_log_pm_cpu_Albany.txt
cp biweekly_log_pm_cpu_Compass.txt /global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial/logs/biweekly_log_pm_cpu_Compass.txt
cp biweekly_log_pm_cpu_MALI.txt /global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial/logs/biweekly_log_pm_cpu_MALI.txt
cp biweekly_log_pm_cpu_Trilinos.txt /global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial/logs/biweekly_log_pm_cpu_Trilinos.txt

chmod -R 2770 /global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial
chown -R fanssie /global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/serial