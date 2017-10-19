#!/bin/bash

echo ""
echo "Extracting norm ref soln for component $2 from file $1..."
if [ $2 == all ]
  then 
    grep "All dofs, |f|_2 (norm ref soln)" $1 >& b
else
  grep "Dof = $2, |f|_2 (norm ref soln)" $1 >& b
fi
if [ $2 = 0 ] 
  then
    sed -e 's/Dof = 0, |f|_2 (norm ref soln)://g' b
elif [ $2 = 1 ]
  then
    sed -e 's/Dof = 1, |f|_2 (norm ref soln)://g' b
elif [ $2 = 2 ]
  then
    sed -e 's/Dof = 2, |f|_2 (norm ref soln)://g' b
elif [ $2 = all ]
  then
    sed -e 's/All dofs, |f|_2 (norm ref soln)://g' b
fi
rm b 
echo "...done!"
