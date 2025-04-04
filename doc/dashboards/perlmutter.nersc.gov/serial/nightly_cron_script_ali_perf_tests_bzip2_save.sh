#!/bin/csh

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-serial
cd $BASE_DIR/build/Testing/Temporary
ln -s $BASE_DIR/repos/ali-perf-tests/jupyter_nb/scripts/ctest2json.py 
python ctest2json.py >& ctest2json.out  
/usr/bin/bzip2 LastTest_*  
cd $BASE_DIR
cp build/Testing/Temporary/LastTest_*bz2 saved/
cp build/Testing/Temporary/ctest*json saved/
cd $BASE_DIR/repos/ali-perf-data/
git pull
cp $BASE_DIR/saved/* $BASE_DIR/repos/ali-perf-data/ali/pm_serial_data
cd $BASE_DIR/repos/ali-perf-data/ali/pm_serial_data
now=$(date +"%Y%m%d")
FILE=ctest-$now.json
git add $FILE
FILE2=LastTest_$now-0100.log.bz2
git add $FILE2
git commit -m "Adding Perlmutter-serial ALI nightly performance tests." 
git push