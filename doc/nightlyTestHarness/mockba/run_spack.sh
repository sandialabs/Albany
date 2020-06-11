#!/bin/bash 

rm -rf run_spack.out 
rm -rf spack_albany.out 
rm -rf spack_libiconv.out 
rm -rf spack 
rm -rf spack_ctest.out 
rm -rf results_spack 
rm -rf /tmp/ikalash/spack*
 
echo "Cloning spack repo..."
git clone git@github.com:SNLComputation/spack.git
cd spack
git checkout albany
echo "...done."
#change where build happens from tempdir 
#sed -i 's,$tempdir/$user,/extradrive,g' etc/spack/defaults/config.yaml
export http_proxy=http://wwwproxy.sandia.gov:80
export https_proxy=http://wwwproxy.sandia.gov:80
. share/spack/setup-env.sh 
echo "Starting spack build..."
spack --insecure install --dirty --keep-stage libiconv >& spack_libiconv.out
spack --insecure install --dirty --keep-stage albany >& spack_albany.out 
cd /home/ikalash/nightlySpackBuild
mv spack/spack_albany.out .
mv spack/spack_libiconv.out .
echo "...done."
cd spack
spack cd albany
cd ../spack-build
echo "Starting running of tests..."
PWD=`pwd`
echo "  spack build dir = " $PWD
ctest -V >& spack_ctest.out 
mv spack_ctest.out /home/ikalash/nightlySpackBuild/spack_ctest.out 
echo "...done."
cd /home/ikalash/nightlySpackBuild
bash process_results_spack.sh 
bash send_email_spack.sh 
