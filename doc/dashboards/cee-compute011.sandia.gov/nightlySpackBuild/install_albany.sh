

cd spack
source sems-gcc-modules.sh 
export http_proxy=http://proxy.sandia.gov:80
export https_proxy=http://proxy.sandia.gov:80
. share/spack/setup-env.sh
#spack compiler find
#cp ../config.yaml etc/spack/defaults/config.yaml
spack --insecure install --dirty --keep-stage albany@develop%gcc@10.1.0+mpas~py+unit_tests >& spack_albany.out
