BASE_DIR=/lustre/orion/cli193/scratch/mcarlson/testingCDashFrontier-rocm
DEPLOY_DIR=/lustre/orion/cli193/proj-shared/automated_testing/rocm

# build trilinos and albany
source ${BASE_DIR}/frontier_gpu_modules.sh >& ${BASE_DIR}/modules_albany.out
bash ${BASE_DIR}/nightly_cron_script_trilinos_frontier.sh
bash ${BASE_DIR}/nightly_cron_script_albany_frontier.sh

sbatch test_albany_sbatch.sh
