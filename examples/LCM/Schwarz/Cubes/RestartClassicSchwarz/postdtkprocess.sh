

if [ ! $3 ] ; then
    echo "This function requires 3 arguments: load step # (int), schwarz step # (int), cube # (int)";
    exit
fi

step=$1
schwarz_iter=$2
cube=$3


#change time stamp in cube"$cube"_in_load"$step"_schwarz"$schwarz_iter".exo to 0 to be consistent with 
#DTK output target file which will have a time stamp of 0
ncap2 -s 'time_whole=0*time_whole' cube"$cube"_in_load"$step"_schwarz"$schwarz_iter".exo tmp.exo
mv tmp.exo cube"$cube"_in_load"$step"_schwarz"$schwarz_iter".exo
#merge target_cube"$cube"_out_load"$step"_schwarz"$schwarz_iter".exo file with cube"$cube"_in_load"$step"_schwarz"$schwarz_iter".exo
#The following produces a file ejoin-out.e
ejoin target_cube"$cube"_out_load"$step"_schwarz"$schwarz_iter".exo cube"$cube"_in_load"$step"_schwarz"$schwarz_iter".exo >& ejoin.out
rm ejoin.out
#rename ejoin-out.e to cube"$cube"_restart_load"$step"_schwarz"$schwarz_iter".exo  
mv ejoin-out.e cube"$cube"_restart_load"$step"_schwarz"$schwarz_iter".exo
