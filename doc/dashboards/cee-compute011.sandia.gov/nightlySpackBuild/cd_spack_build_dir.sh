echo $PWD >& a 
sed -i "s,/scratch/albany/nightlySpackBuild/spack-stage/ikalash/spack-stage-albany-develop-,,g" a 
cut -c-7 a >& b 
echo "cd spack-build-" >& c 
cat c b >& cb
perl -p -i -e 's/\R//g;' cb
source cb

