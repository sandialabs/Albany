
set terminal pdf enhanced font "Times-Roman,24" size 12in, 8in 
set output "crack_test_nonlocal_top_plot.pdf"
set xlabel "Position along Length of Bar" font "Times-Roman,32"
set ylabel "Solution" font "Times-Roman,32"
set xrange [-1.5:1.5]
set yrange [-0.006:0.006]
set key bottom right
plot "OBC_Crack_Analysis_pd_top.txt" using 1:2 with points pt 2 ps 6 lw 20 lc 2 title "Nonlocal Model Initial Values", \
     "OBC_Crack_Analysis_pd_top.txt" using 1:3 with points pt 7 ps 4 lc 3 title "Nonlocal Model Final Values"

set terminal pdf enhanced font "Times-Roman,24" size 12in, 8in 
set output "crack_test_local_top_plot.pdf"
set xlabel "Position along Length of Bar" font "Times-Roman,32"
set ylabel "Solution" font "Times-Roman,32"
set xrange [-1.5:1.5]
set yrange [-0.006:0.006]
set key bottom right
plot "OBC_Crack_Analysis_fem_top.txt" using 1:2 with points pt 2 ps 6 lw 20 lc 2 title "Local Model Initial Values", \
     "OBC_Crack_Analysis_fem_top.txt" using 1:3 with points pt 7 ps 4 lc 3 title "Local Model Final Values"

set terminal pdf enhanced font "Times-Roman,24" size 12in, 8in 
set output "crack_test_nonlocal_bottom_plot.pdf"
set xlabel "Position along Length of Bar" font "Times-Roman,32"
set ylabel "Solution" font "Times-Roman,32"
set xrange [-1.5:1.5]
set yrange [-0.006:0.006]
set key bottom right
plot "OBC_Crack_Analysis_pd_bottom.txt" using 1:2 with points pt 2 ps 6 lw 20 lc 2 title "Nonlocal Model Initial Values", \
     "OBC_Crack_Analysis_pd_bottom.txt" using 1:3 with points pt 7 ps 4 lc 3 title "Nonlocal Model Final Values"

set terminal pdf enhanced font "Times-Roman,24" size 12in, 8in 
set output "crack_test_local_bottom_plot.pdf"
set xlabel "Position along Length of Bar" font "Times-Roman,32"
set ylabel "Solution" font "Times-Roman,32"
set xrange [-1.5:1.5]
set yrange [-0.006:0.006]
set key bottom right
plot "OBC_Crack_Analysis_fem_bottom.txt" using 1:2 with points pt 2 ps 6 lw 20 lc 2 title "Local Model Initial Values", \
     "OBC_Crack_Analysis_fem_bottom.txt" using 1:3 with points pt 7 ps 4 lc 3 title "Local Model Final Values"
