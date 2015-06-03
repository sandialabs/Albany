
# cross section of specimen = 1.0
# initial length of specimen = 1.0

set terminal pdf enhanced font "Times-Roman,24" size 12in, 8in 
set output "RubiksCubeForceDisplacement.pdf"
set xlabel "Engineering Strain" font "Times-Roman,32"
set ylabel "Engineering Stress" font "Times-Roman,32"
#set xrange [0.0:0.015]
#set key at 0.01, 1.18
set key font ",16"
plot "RubiksCube_force_displacement.txt" using 1:2 with points pt 7 ps 4 lc 2 notitle
