#!/bin/csh

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

BASE_DIR=/lustre/orion/cli193/scratch/jwatkins/nightlyCDashFrontier
SRC_DIR=/ccs/proj/cli193/automated_testing
cd $BASE_DIR/build/Testing/Temporary
ln -s $SRC_DIR/repos/ali-perf-tests/jupyter_nb/scripts/ctest2json.py 
python ctest2json.py >& ctest2json.out  
/usr/bin/bzip2 LastTest_*  
cd $BASE_DIR
cp build/Testing/Temporary/LastTest_*bz2 saved/
cp build/Testing/Temporary/ctest*json saved/
cd $SRC_DIR/repos/ali-perf-data/
#git pull # remote work doesn't seem to work on compute node
cp $BASE_DIR/saved/* $SRC_DIR/repos/ali-perf-data/ali/frontier_data
cd $SRC_DIR/repos/ali-perf-data/ali/frontier_data
now=$(date +"%Y%m%d")
FILE=ctest-$now.json
git add $FILE
FILE2=LastTest_$now-0100.log.bz2
git add $FILE2
git commit -m "Adding Frontier ALI nightly performance tests." 
# git push # remote work doesn't seem to work on compute node
