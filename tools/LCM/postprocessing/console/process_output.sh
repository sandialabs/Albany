
#!/bin/bash
#throw error if no argument was supplied
if [ ! $1 ] ; then 
    echo "No output file name supplied!  This file needs to be run with output file from Albany run.";
    exit  
fi
echo "Starting data processing from file" $1 "..."
grep "CONVERGED" $1 >& linearsolves
sed -i -e 's/.*CONVERGED" in//g' linearsolves
sed -i -e 's/iterations with total CPU time of//g' linearsolves
sed -i -e 's/sec//g' linearsolves
wc -l linearsolves
grep "||F||" $1 >& nonlinsolves
sed -i -e 's/||F|| =//g' nonlinsolves
sed -i -e 's/step =//g' nonlinsolves
sed -i -e 's/dx =//g' nonlinsolves
wc -l nonlinsolves
sed -i -e 's/(Converged!)/\n1e7 0 0/g' nonlinsolves
grep "Total Time" $1
python calcmeanlinsolves.py
python calcnonlinsolves.py 
