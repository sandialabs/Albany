#!/bin/bash
### use dnf list 
echo ">>> checking dnf packages <<<"
error="NONE"
for pkg in \
blas \
blas-devel \
boost \
boost-devel \
boost-openmpi \
boost-openmpi-devel \
boost-static \
cmake \
environment-modules \
gcc-c++ \
git \
hdf5 \
hdf5-devel \
hdf5-openmpi \
hdf5-openmpi-devel \
hdf5-static \
hwloc-devel \
hwloc-libs \
lapack \
lapack-devel \
matio \
matio-devel \
netcdf \
netcdf-devel \
netcdf-openmpi \
netcdf-openmpi-devel \
netcdf-static \
openmpi \
openmpi-devel \
yaml-cpp \
yaml-cpp-devel; do
  query=`dnf list $pkg |& tail -n1`
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
  git clone git@github.com:trilinos/Trilinos.git Trilinos
else
  echo ">>> Trilinos exists, freshening it <<<"
  (cd Trilinos; git pull)
fi
if [ ! -d Albany ]; then
  git clone git@github.com:gahansen/Albany.git Albany
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

if [[ -z $LCM_DIR ]]; then
  echo "ERROR: Top level LCM directory not defined."
  exit
fi
if [[ $MODULEPATH != "$LCM_DIR/Albany/doc/LCM/modulefiles" ]]; then
  echo "ERROR: Path to LCM modules set incorrectly."
  echo "MODULEPATH: $MODULEPATH"
  exit
fi

NP=`nproc`
toolchain="gcc"
machinetype="serial"
#buildtype="debug"
buildtype="release"

module load lcm/fedora
module load $machinetype-$toolchain-$buildtype

for target in trilinos albany; do
 dir=${target}-build-${machinetype}-${toolchain}-${buildtype}
 if [ ! -d $dir ]; then
   echo "!!! $dir exists !!!"
 fi
 echo ">>> building ${target}-${machinetype}-${toolchain}-${buildtype} with ${NP} processes <<<"
 ./clean-config-build.sh ${target} $NP >& ${target}_build.log
done

dir="albany-build-${machinetype}-${toolchain}-${buildtype}"
if [ -e $dir/src/Albany ]; then
  echo "=== build in $dir successful ==="
else
  echo "!!! unsuccessful build in $dir, see logs !!!"
fi

echo "..........................................................................................."
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
