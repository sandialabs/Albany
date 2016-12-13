module purge
module load gnu/4.9.2
module load openmpi-gnu/1.8
module load python/2.7

# Building Trilinos, add environmental variable and path
export REMOTE=/gscratch/jwfoulk/albany
export REMOTE_EXEC=/gscratch/jwfoulk/albany_builds
export LD_LIBRARY_PATH=/opt/python-2.7/lib:$LD_LIBRARY_PATH

# Using Trilinos
LD_LIBRARY_PATH=$REMOTE/trilinos-install-gcc-release/lib:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=$REMOTE/lib:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=$REMOTE/lib64:$LD_LIBRARY_PATH
PATH=$REMOTE/trilinos-install-gcc-release/bin:$PATH

echo "Build directory: " $REMOTE
echo "Executable directory:" $REMOTE_EXEC

# Blow out install directories
rm -f -r $REMOTE/trilinos-install-gcc-release/
rm -f -r $REMOTE/albany-build-gcc-release/

# Pull new repositories
cd $REMOTE/src/Albany
git pull
cd $REMOTE/src/Trilinos
git pull

# Build a fresh flavor of Trilinos
rm -f -r $REMOTE/src/build
mkdir $REMOTE/src/build
cp $REMOTE/src/Albany/doc/skybridge-chama/trilinos-config.sh $REMOTE/src/build/
cd $REMOTE/src/build
./trilinos-config.sh
make -j 8 
make install -j 8

# Build a fresh flavor of Albany
cd $REMOTE
mkdir albany-build-gcc-release
cp $REMOTE/src/Albany/doc/skybridge-chama/albany-config.sh $REMOTE/src/albany-build-gcc-release
cd albany-build-gcc-release
./albany-config.sh
make -j 8

# Make a directory for the build
export DATE=$(date +%m-%d-%Y")
mkdir $REMOTE_EXEC/$DATE
chmod a+rx $REMOTE_EXEC/$DATE 

# Copy the executables and libraries
cp -r $REMOTE/albany-build-gcc-release $REMOTE_EXEC/$DATE
cp -r $REMOTE/lib $REMOTE_EXEC/$DATE
cp -r $REMOTE/lib64 $REMOTE_EXEC/$DATE
cp -r $REMOTE/trilinos-install-gcc-release $REMOTE_EXEC/$DATE

# open up premissions
cd $REMOTE_EXEC/$DATE
find albany-build-gcc-release -perm /a=x -exec chmod a+rx {} \;
find lib -perm /a=x -exec chmod a+rx {} \; 
find lib64 -perm /a=x -exec chmod a+rx {} \;
find trilinos-install-gcc-release -perm /a=x -exec chmod a+rx {} \;




