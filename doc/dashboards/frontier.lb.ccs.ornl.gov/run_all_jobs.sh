BASE_DIR=/lustre/orion/cli193/scratch/mcarlson/testingCDashFrontier-rocm
DEPLOY_DIR=/lustre/orion/cli193/proj-shared/automated_testing/rocm

# Update albany repo to get most recent changes
cd ${BASE_DIR}/repos/Albany
git pull
cd ${BASE_DIR}

# load modules
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


