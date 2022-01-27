#!/bin/sh

cd /nightlyCDash

#unset http_proxy
#unset https_proxy

export https_proxy="http://proxy.ca.sandia.gov:80"
export http_proxy="http://proxy.ca.sandia.gov:80"

LOG_FILE=/nightlyCDash/nightly_log_alegra-xfem.txt
eval "env TEST_DIRECTORY=/nightlyCDash SCRIPT_DIRECTORY=/nightlyCDash ctest -VV -S /nightlyCDash/ctest_nightly_alegra-xfem.cmake" > $LOG_FILE 2>&1

LOG_FILE=/nightlyCDash/nightly_log_alegra-xfem_eti.txt
eval "env TEST_DIRECTORY=/nightlyCDash SCRIPT_DIRECTORY=/nightlyCDash ctest -VV -S /nightlyCDash/ctest_nightly_alegra-xfem_eti.cmake" > $LOG_FILE 2>&1

# Copy a basic installation to /projects/albany for those who like a nightly
# build.
#cp -r build/TrilinosInstall/* /projects/albany/trilinos/nightly/;
#chmod -R a+X /projects/albany/trilinos/nightly;
#chmod -R a+r /projects/albany/trilinos/nightly;
