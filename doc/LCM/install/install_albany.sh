#!/bin/bash
### use yum list 
echo ">>> checking yum packages <<<"
error="NONE"
for pkg in \
blas blas-devel lapack lapack-devel openmpi openmpi-devel \
netcdf netcdf-devel netcdf-static netcdf-openmpi netcdf-openmpi-devel \
hdf5 hdf5-devel hdf5-static hdf5-openmpi hdf5-openmpi-devel \
boost boost-devel boost-static boost-openmpi boost-openmpi-devel \
matio matio-devel libX11 libX11-devel cmake gcc-c++ git; do
  query=`yum list $pkg |& tail -n1`
  if [ ${query:0:5} == "Error" ]; then
    echo "MISSING $pkg"
    error="MISSING"
  else
    echo "ok      $pkg"
  fi
done
if [ ! $error == "NONE" ]; then
  exit
fi

if [ ! -d Trilinos ]; then
  #git clone git@github.com:trilinos/trilinos.git Trilinos
  git clone software.sandia.gov:/space/git/Trilinos Trilinos
  #git clone https://github.com/trilinos/trilinos.git Trilinos
else
  echo ">>> Trilinos exists, freshening it <<<"
  (cd Trilinos; git pull)
fi
if [ ! -d Albany ]; then
  git clone git@github.com:gahansen/Albany.git Albany
  #git clone https://github.com/gahansen/Albany.git Albany
else
  echo ">>> Albany exists, freshening it <<<"
  (cd Albany; git pull)
fi

ln -sf Albany/doc/LCM/build/*.sh .
ln -sf build.sh clean.sh
ln -sf build.sh config.sh
ln -sf build.sh test.sh
ln -sf build.sh clean-config.sh
ln -sf build.sh clean-config-build.sh
ln -sf build.sh clean-config-build-test.sh
ln -sf build.sh config-build.sh
ln -sf build.sh config-build-test.sh

echo "NOTE for testing: change FROM & TO email addresses in env-single.sh"

if [[ $PATH != *"/usr/lib64/openmpi/bin"* ]]; then
  echo "!!! add /usr/lib64/openmpi/bin to PATH !!!"
  echo "PATH: $PATH"
  exit
fi
if [[ $LD_LIBRARY_PATH != *"/usr/lib64/openmpi/lib"* ]]; then
  echo "!!! add /usr/lib64/openmpi/lib to LD_LIBRARY_PATH !!!"
  echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
  exit
fi

NP=`nproc`
toolchain="gcc"
#buildtype="debug"
buildtype="release"

for target in trilinos albany; do
 dir=${target}-build-${toolchain}-${buildtype}
 if [ ! -d $dir ]; then
   echo "!!! $dir exists !!!"
 fi
 echo ">>> building ${target}-${toolchain}-${buildtype} with ${NP} processes <<<"
 ./clean-config-build.sh ${target} ${toolchain} ${buildtype} $NP >& ${target}_build.log
done

if [ -e albany-build-${toolchain}-${buildtype}/src/Albany ]; then
  echo "=== build successful ==="
else
  echo "!!! unsuccessful build, see logs !!!"
fi

echo "to ensure proper git behavior as a developer/commiter add the following to your .gitconfig:"
echo " \
[branch]
        autosetuprebase = always
[color]
        ui = true
[core]
        whitespace = -trailing-space,-space-before-tab
        preloadingindex = true
        preloadindex = true

[branch \"master\"]
        rebase = true
"

