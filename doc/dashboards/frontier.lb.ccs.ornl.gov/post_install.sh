cd /lustre/orion/cli193/scratch/jwatkins/nightlyCDashFrontier
DEPLOY_DIR=/ccs/proj/cli193/automated_testing/rocm
cp test_log_frontier_Albany.txt ${DEPLOY_DIR}/logs/test_log_frontier_Albany.txt
cp build_log_frontier_Albany.txt ${DEPLOY_DIR}/logs/build_log_frontier_Albany.txt
cp build_log_frontier_Trilinos.txt ${DEPLOY_DIR}/logs/build_log_frontier_Trilinos.txt
chmod -R 2770 ${DEPLOY_DIR}
chown -R :cli193 ${DEPLOY_DIR}
