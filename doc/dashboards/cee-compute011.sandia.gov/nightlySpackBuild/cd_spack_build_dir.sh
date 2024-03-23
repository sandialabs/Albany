echo $PWD >& a 
sed -i "s,/tmp/ikalash/spack-stage/spack-stage-albany-develop-,,g" a 
cut -c-7 a >& b 
echo "cd spack-build-" >& c 
cat c b >& cb
perl -p -i -e 's/\R//g;' cb
source cb

