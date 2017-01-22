
# cross section of specimen = 1.0
# initial length of specimen = 1.0

set terminal pdf enhanced font "Times-Roman,24" size 12in, 8in 
set output "SingleSlipForceDisplacement.pdf"
set xlabel "Engineering Strain (m/m)" font "Times-Roman,32"
set ylabel "Engineering Stress (GPa)" font "Times-Roman,32"
set xrange [0.0:0.015]
set key at 0.01, 1.18
set key font ",16"
plot "SingleSlip_Explicit_block_1_force_displacement.txt" using 1:2 with points pt 7 ps 4 lc 2 title "Explicit integration, Crystal plane rotated 90 degrees", \
     "SingleSlip_Explicit_block_2_force_displacement.txt" using 1:2 with points pt 7 ps 4 lc 3 title "Explicit integration, Crystal plane rotated 45 degrees", \
     "SingleSlip_Implicit_block_1_force_displacement.txt" using 1:2 with points pt 7 ps 4 lc 4 title "Implicit integration, Crystal plane rotated 90 degrees", \
     "SingleSlip_Implicit_block_2_force_displacement.txt" using 1:2 with points pt 7 ps 4 lc 5 title "Implicit integration, Crystal plane rotated 45 degrees"
