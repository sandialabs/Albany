
# cross section of specimen = 1.0
# initial length of specimen = 1.0

set terminal pdf enhanced font "Times-Roman,24" size 12in, 8in 
set output "SingleSlipPlaneForceDisplacement.pdf"
set xlabel "Engineering Strain (m/m)" font "Times-Roman,32"
set ylabel "Engineering Stress (GPa)" font "Times-Roman,32"
set xrange [0.0:0.015]
set key bottom right
plot "block_1_force_displacement.txt" using 1:2 with points pt 7 ps 4 lc 2 title "Crystal plane rotated 90 degrees", \
     "block_2_force_displacement.txt" using 1:2 with points pt 7 ps 4 lc 3 title "Crystal plane rotated 45 degrees"
