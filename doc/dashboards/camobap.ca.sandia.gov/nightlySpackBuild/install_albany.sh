

cd spack
export http_proxy=http://proxy.sandia.gov:80
export https_proxy=http://proxy.sandia.gov:80
. share/spack/setup-env.sh
#spack compiler find
#cp ../config.yaml etc/spack/defaults/config.yaml
#spack --insecure install --dirty --keep-stage albany@develop%gcc@11.1.0+mpas+py+optimization+mesh_depends_on_params+omegah >& spack_albany.out
spack --insecure install --dirty --keep-stage albany@develop%gcc@11.1.0+mpas+optimization+mesh_depends_on_params+omegah >& spack_albany.out
