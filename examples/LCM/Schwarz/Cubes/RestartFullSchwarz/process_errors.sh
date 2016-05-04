
#!/bin/bash

if [ ! $1 ] ; then
    echo "This function requires 1 arguments: # load steps (int)";
    exit
fi

for (( step=0; step<$1; step++ )); do
  file_name_star=error_load"$step"_schwarz*
  file_name=./error_load"$step"_schwarz
  find . -type f -name "$file_name_star" >& error_load"$step"_filenames
  find . -type f -name "$file_name_star" -exec cat {} + >& error_load"$step"_values
  sed -i "s,$file_name,,g" error_load"$step"_filenames
done
