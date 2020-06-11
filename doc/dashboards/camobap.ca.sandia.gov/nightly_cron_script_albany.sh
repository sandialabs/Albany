#!/bin/sh

cd /nightlyCDash

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib64/openmpi/lib

#unset http_proxy
#unset https_proxy

export https_proxy="https://wwwproxy.ca.sandia.gov:80"
export http_proxy="http://wwwproxy.ca.sandia.gov:80"

LOG_FILE=/nightlyCDash/nightly_log.txt

eval "env TEST_DIRECTORY=/nightlyCDash SCRIPT_DIRECTORY=/nightlyCDash ctest -VV -S /nightlyCDash/ctest_nightly_albany.cmake" > $LOG_FILE 2>&1

# Copy a basic installation to /projects/albany for those who like a nightly
# build.
#cp -r build/TrilinosInstall/* /projects/albany/trilinos/nightly/;
#chmod -R a+X /projects/albany/trilinos/nightly;
#chmod -R a+r /projects/albany/trilinos/nightly;
