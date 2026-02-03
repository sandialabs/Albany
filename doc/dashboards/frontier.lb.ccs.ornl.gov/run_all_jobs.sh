BASE_DIR=/lustre/orion/cli193/scratch/jwatkins/nightlyCDashFrontier
DEPLOY_DIR=/ccs/proj/cli193/automated_testing/rocm

# Update Trilinos repo to get most recent changes
cd ${BASE_DIR}/repos/Trilinos
git pull

# Update albany repo to get most recent changes
cd ${BASE_DIR}/repos/Albany
git pull

# Update E3SM repo to get most recent changes
cd ${BASE_DIR}/repos/E3SM
git pull
git submodule sync --recursive && git submodule update --init --recursive

# Update ali-perf-test repo to get most recent changes
cd ${BASE_DIR}/repos/ali-perf-tests
git pull

# Update ali-perf-data repo to get most recent changes
cd ${BASE_DIR}/repos/ali-perf-data
git pull

# load modules
cd ${BASE_DIR}
source ${BASE_DIR}/frontier_gpu_modules.sh >& ${BASE_DIR}/modules_albany.out

# build trilinos and albany
bash ${BASE_DIR}/nightly_cron_script_trilinos_frontier.sh
bash ${BASE_DIR}/nightly_cron_script_albany_frontier.sh slfad none
bash ${BASE_DIR}/nightly_cron_script_albany_frontier.sh sfad 12
bash ${BASE_DIR}/nightly_cron_script_albany_frontier.sh sfad 24

# build MALI
bash ${BASE_DIR}/nightly_cron_script_mali_frontier.sh

# run regression tests
sbatch test_albany_sbatch.sh

# run performance tests
sbatch perf_test_albany_sbatch.sh

cp build_log_frontier_Albany.txt ${DEPLOY_DIR}/logs/build_log_frontier_Albany_slfad.txt
cp build_log_frontier_Albany.txt ${DEPLOY_DIR}/logs/build_log_frontier_Albany_sfad12.txt
cp build_log_frontier_Albany.txt ${DEPLOY_DIR}/logs/build_log_frontier_Albany_sfad24.txt
cp build_log_frontier_Trilinos.txt ${DEPLOY_DIR}/logs/build_log_frontier_Trilinos.txt
cp build_log_frontier_MALI.txt ${DEPLOY_DIR}/logs/build_log_frontier_MALI.txt

chmod -R 2770 ${DEPLOY_DIR}
chown -R :cli193 ${DEPLOY_DIR}


