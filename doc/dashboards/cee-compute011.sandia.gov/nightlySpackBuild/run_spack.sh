

cd spack
export http_proxy=http://proxy.sandia.gov:80
export https_proxy=http://proxy.sandia.gov:80
. share/spack/setup-env.sh
module load cde/dev/compiler/gcc/7.2.0
spack compiler find
cp ../config.yaml etc/spack/defaults/config.yaml
spack --insecure install --dirty --keep-stage libtool%gcc@7.2.0 >& spack_libtool.out
sleep 5m
spack --insecure install --dirty --keep-stage diffutils%gcc@7.2.0 >& spack_diffutils.out
sleep 5m
spack --insecure install --dirty --keep-stage xz%gcc@7.2.0 >& spack_xz.out
sleep 5m
spack --insecure install --dirty --keep-stage albany%gcc@7.2.0+mpas >& spack_albany.out
#spack cd albany >& /projects/albany/nightlySpackBuild/spack_cd_albany.out
cp cd_spack_build_dir.sh /scratch/albany/nightlySpackBuild/spack-stage/ikalash/spack-stage-albany-develop* 
cd /scratch/albany/nightlySpackBuild/spack-stage/ikalash/spack-stage-albany-develop* >& /projects/albany/nightlySpackBuild/spack_cd_albany.out
source cd_spack_build_dir.sh
ctest -V >& spack_ctest.out
mv spack_ctest.out /projects/albany/nightlySpackBuild/spack_ctest.out
cd /projects/albany/nightlySpackBuild
source /projects/albany/nightlySpackBuild/postproc_spack.sh
