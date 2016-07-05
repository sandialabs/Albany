
set terminal pdf enhanced font "Times-Roman,24" size 12in, 8in 
set output "patch_test_nonlocal_plot.pdf"
set xlabel "Position along Length of Bar" font "Times-Roman,32"
set ylabel "Solution" font "Times-Roman,32"
set xrange [-0.9:0.2]
set yrange [-0.0006:0.0006]
set key bottom right
plot "OBC_PatchTest_Analysis_pd.txt" using 1:2 with points pt 2 ps 6 lw 20 lc 2 title "Nonlocal Model Initial Values", \
     "OBC_PatchTest_Analysis_pd.txt" using 1:3 with points pt 7 ps 4 lc 3 title "Nonlocal Model Final Values"

set terminal pdf enhanced font "Times-Roman,24" size 12in, 8in 
set output "patch_test_local_plot.pdf"
set xlabel "Position along Length of Bar" font "Times-Roman,32"
set ylabel "Solution" font "Times-Roman,32"
set xrange [-0.2:0.9]
set yrange [-0.0006:0.0006]
set key bottom right
plot "OBC_PatchTest_Analysis_fem.txt" using 1:2 with points pt 2 ps 6 lw 20 lc 2 title "Local Model Initial Values", \
     "OBC_PatchTest_Analysis_fem.txt" using 1:3 with points pt 7 ps 4 lc 3 title "Local Model Final Values"

set terminal pdf enhanced font "Times-Roman,24" size 12in, 8in 
set output "patch_test_combined_plot.pdf"
set xlabel "Position along Length of Bar" font "Times-Roman,32"
set ylabel "Solution" font "Times-Roman,32"
set xrange [-0.9:0.9]
set yrange [-0.0006:0.0006]
set key bottom right
plot "OBC_PatchTest_Analysis_all.txt" using 1:2 with points pt 2 ps 6 lw 20 lc 2 title "Initial Values", \
     "OBC_PatchTest_Analysis_all.txt" using 1:3 with points pt 7 ps 4 lc 3 title "Final Values"


