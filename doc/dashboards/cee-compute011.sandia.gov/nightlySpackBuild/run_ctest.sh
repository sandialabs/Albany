cp cd_spack_build_dir.sh /tmp/ikalash/spack-stage/spack-stage-albany-develop* 
cd /tmp/ikalash/spack-stage/spack-stage-albany-develop* >& /projects/albany/nightlySpackBuild/spack_cd_albany.out
source cd_spack_build_dir.sh
ctest --timeout 1200 -V >& spack_ctest.out
mv spack_ctest.out /projects/albany/nightlySpackBuild/spack_ctest.out
cd /projects/albany/nightlySpackBuild
source /projects/albany/nightlySpackBuild/postproc_spack.sh
