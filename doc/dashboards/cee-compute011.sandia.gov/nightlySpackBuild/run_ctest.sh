cp cd_spack_build_dir.sh /scratch/albany/nightlySpackBuild/spack-stage/ikalash/spack-stage-albany-develop* 
cd /scratch/albany/nightlySpackBuild/spack-stage/ikalash/spack-stage-albany-develop* >& /projects/albany/nightlySpackBuild/spack_cd_albany.out
source cd_spack_build_dir.sh
ctest -V >& spack_ctest.out
mv spack_ctest.out /projects/albany/nightlySpackBuild/spack_ctest.out
cd /projects/albany/nightlySpackBuild
source /projects/albany/nightlySpackBuild/postproc_spack.sh
