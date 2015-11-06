
set terminal pdf enhanced font "Times-Roman,24" size 12in, 8in 
set output "TensileBarPDPS.pdf"
set xlabel "Position along Length of Bar" font "Times-Roman,32"
set ylabel "Displacement" font "Times-Roman,32"
set xrange [-1.5:1.5]
set yrange [-0.02:0.02]
set key bottom right
plot "TensileBarPDPS_ps.txt" using 1:2 with points pt 7 ps 4 lc 3 title "Peridynamic Partial Stress", \
     "TensileBarPDPS_pd.txt" using 1:2 with points pt 2 ps 6 lw 20 lc 2 title "Meshfree Peridynamics", \
     "linear_solution.txt" using 1:2 with lines lw 2 lc -1 title "Expected Linear Solution"


