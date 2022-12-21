nvcc --version >& cuda.txt
grep "Build" cuda.txt >& cuda2.txt
mv cuda2.txt cuda.txt
sed -i 's/Build //g' cuda.txt 
sed -i 's/\//\t/g' cuda.txt 
cat cuda.txt | awk '{ print $1 }'

