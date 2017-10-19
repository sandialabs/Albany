#!/bin/bash

echo ""
echo "Extracting absolute error for component $2 from file $1..."
if [ $2 == all ]
  then 
    grep "All dofs, |e|_2 (abs error)" $1 >& b
else
  grep "Dof = $2, |e|_2 (abs error)" $1 >& b
fi
if [ $2 = 0 ] 
  then
    sed -e 's/Dof = 0, |e|_2 (abs error)://g' b
elif [ $2 = 1 ]
  then
    sed -e 's/Dof = 1, |e|_2 (abs error)://g' b
elif [ $2 = 2 ]
  then
    sed -e 's/Dof = 2, |e|_2 (abs error)://g' b
elif [ $2 = all ]
  then
    sed -e 's/All dofs, |e|_2 (abs error)://g' b
fi
rm b 
echo "...done!"
