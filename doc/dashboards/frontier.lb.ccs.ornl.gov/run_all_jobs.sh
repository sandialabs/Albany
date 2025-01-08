BASE_DIR=/lustre/orion/cli193/scratch/mcarlson/testingCDashFrontier-rocm
DEPLOY_DIR=/lustre/orion/cli193/proj-shared/automated_testing/rocm

# get albany repo once (instead of for each build)
mkdir ${BASE_DIR}/repos
git clone https://github.com/sandialabs/Albany.git ${BASE_DIR}/repos/Albany

# load modules
source ${BASE_DIR}/frontier_gpu_modules.sh >& ${BASE_DIR}/modules_albany.out

# build trilinos and albany
bash ${BASE_DIR}/nightly_cron_script_trilinos_frontier.sh
bash ${BASE_DIR}/nightly_cron_script_albany_frontier.sh slfad none
bash ${BASE_DIR}/nightly_cron_script_albany_frontier.sh sfad 12
bash ${BASE_DIR}/nightly_cron_script_albany_frontier.sh sfad 24

sbatch test_albany_sbatch.sh
