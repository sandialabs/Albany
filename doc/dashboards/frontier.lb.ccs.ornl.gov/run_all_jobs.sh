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

sbatch test_albany_sbatch.sh
