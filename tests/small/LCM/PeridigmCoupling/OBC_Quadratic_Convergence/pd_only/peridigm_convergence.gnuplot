
f(x) = m1*x + b1
m1 = 0.1; b1 = 1;
fit [-2.0:0.0] f(x) "error_data.txt" u ($4):($5) via m1, b1

set terminal pdf enhanced font "Times,32" size 12in, 8in
set output "pd_only_convergence.pdf"
set title "Peridynamics Only"
set xlabel "log(h)"
set ylabel "log(error)"
set label at -1.72,3.325 "Rate = X.XX"
plot "error_data.txt" using 4:5 with linespoints lw 10 lc -1 notitle
