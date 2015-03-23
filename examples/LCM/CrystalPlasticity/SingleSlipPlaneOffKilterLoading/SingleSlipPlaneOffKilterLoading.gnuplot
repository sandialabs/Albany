
set terminal pdf enhanced font "Times-Roman,24" size 12in, 8in 
set output "SingleSlipPlaneOffKilterLoading_MaxPrincipalStress_versus_Time.pdf"
set xlabel "Time (s)" font "Times-Roman,32"
set ylabel "Maximum Principal Stress (GPa)" font "Times-Roman,32"
set key bottom right
plot "max_principal_stress_versus_time.txt" using 1:2 with points pt 7 ps 4 lc 2 title "Crystal plane aligned with loading direction", \
     "max_principal_stress_versus_time.txt" using 1:3 with points pt 7 ps 4 lc 3 title "Crystal plane misaligned with loading direction"
