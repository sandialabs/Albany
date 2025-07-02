#!/bin/sh

cd /nightlyCDash

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

#export https_proxy="http://proxy.ca.sandia.gov:80"
#export http_proxy="http://proxy.ca.sandia.gov:80"
export PATH=$PATH:/tpls/install/ninja/build-cmake

LOG_FILE=/nightlyCDash/nightly_log.txt

bash convert-cmake-to-cdash.sh regular
bash create-new-cdash-cmake-script.sh regular

eval "env TEST_DIRECTORY=/nightlyCDash SCRIPT_DIRECTORY=/nightlyCDash ctest -VV -S /nightlyCDash/ctest_nightly_albany.cmake" > $LOG_FILE 2>&1

# Copy a basic installation to /projects/albany for those who like a nightly
# build.
#cp -r build/TrilinosInstall/* /projects/albany/trilinos/nightly/;
#chmod -R a+X /projects/albany/trilinos/nightly;
#chmod -R a+r /projects/albany/trilinos/nightly;
