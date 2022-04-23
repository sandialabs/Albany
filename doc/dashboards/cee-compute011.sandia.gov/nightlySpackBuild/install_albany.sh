

cd spack
export http_proxy=http://proxy.sandia.gov:80
export https_proxy=http://proxy.sandia.gov:80
. share/spack/setup-env.sh
module load sems-env
module load sems-gcc/9.2.0
spack compiler find
cp ../config.yaml etc/spack/defaults/config.yaml
spack --insecure install --dirty --keep-stage libtool%gcc@9.2.0 >& spack_libtool.out
sleep 5m
spack --insecure install --dirty --keep-stage diffutils%gcc@9.2.0 >& spack_diffutils.out
sleep 5m
spack --insecure install --dirty --keep-stage xz%gcc@9.2.0 >& spack_xz.out
sleep 5m
spack --insecure install --dirty --keep-stage albany%gcc@9.2.0+mpas >& spack_albany.out
