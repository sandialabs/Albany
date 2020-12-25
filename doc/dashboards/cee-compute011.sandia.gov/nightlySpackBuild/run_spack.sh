

cd spack
export http_proxy=http://wwwproxy.sandia.gov:80
export https_proxy=http://wwwproxy.sandia.gov:80
. share/spack/setup-env.sh
spack --insecure install --dirty --keep-stage libtool >& spack_libtool.out
sleep 5m
spack --insecure install --dirty --keep-stage diffutils >& spack_diffutils.out
sleep 5m
spack --insecure install --dirty --keep-stage trilinos >& spack_trilinos.out
sleep 5m
spack --insecure install --dirty --keep-stage albany >& spack_albany.out
spack cd albany >& /projects/albany/nightlySpackBuild/spack_cd_albany.out
cd ../spack-build
ctest -V >& spack_ctest.out
mv spack_ctest.out /projects/albany/nightlySpackBuild/spack_ctest.out
cd /projects/albany/nightlySpackBuild
source /projects/albany/nightlySpackBuild/postproc_spack.sh
