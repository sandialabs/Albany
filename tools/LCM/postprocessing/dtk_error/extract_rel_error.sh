#!/bin/bash

echo "" 
echo "Extracting relative error for component $2 from file $1..."
cp $1 a 
sed -i -e 's,\/,,g' a
if [ $2 = all ]
  then
    grep "All dofs, |e|_2  |f|_2 (rel error)" a >& b
else
  grep "Dof = $2, |e|_2  |f|_2 (rel error)" a >& b
fi
rm a 
if [ $2 = 0 ] 
  then
    sed -e 's/Dof = 0, |e|_2  |f|_2 (rel error)://g' b
elif [ $2 = 1 ]
  then
    sed -e 's/Dof = 1, |e|_2  |f|_2 (rel error)://g' b
elif [ $2 = 2 ]
  then
    sed -e 's/Dof = 2, |e|_2  |f|_2 (rel error)://g' b
elif [ $2 = all ]
  then
    sed -e 's/All dofs, |e|_2  |f|_2 (rel error)://g' b
fi
rm b
echo "...done!"
