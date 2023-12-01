ulimit -c 0

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-cuda

source ${BASE_DIR}/pm_gpu_gnu_modules.sh >& ${BASE_DIR}/modules_albany.out
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDATOOLKIT_HOME}/../../math_libs/${CUDATOOLKIT_VERSION_STRING}/lib64

bash ${BASE_DIR}/nightly_cron_script_trilinos_pm_gpu.sh
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/cuda/builds/TrilinosInstall/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/cuda/builds/TrilinosInstall/lib64

bash ${BASE_DIR}/nightly_cron_script_albany_pm_gpu.sh

cp biweekly_log_pm_gpu_Albany.txt /global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/cuda/logs/biweekly_log_pm_gpu_Albany.txt
cp biweekly_log_pm_gpu_Trilinos.txt /global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/cuda/logs/biweekly_log_pm_gpu_Trilinos.txt

chmod -R 2770 /global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/cuda
chown -R :fanssie /global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/cuda