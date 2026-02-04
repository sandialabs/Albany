BASE_DIR=/pscratch/sd/j/jwatkins/nightlyCDashPerlmutterSerial
cd ${BASE_DIR}
source ${BASE_DIR}/clean-up.sh
mkdir ${BASE_DIR}/repos
git clone https://github.com/sandialabs/Albany.git ${BASE_DIR}/repos/Albany
source ${BASE_DIR}/build_trilinos.sh
sbatch ${BASE_DIR}/build_and_test_albany_slfad.sh
sbatch ${BASE_DIR}/build_and_test_albany_sfad12.sh
sbatch ${BASE_DIR}/build_and_test_albany_sfad24.sh
