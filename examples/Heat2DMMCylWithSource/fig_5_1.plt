set terminal postscript color enhanced
set output "fig_5_1.eps"
#set size 0.5,0.5
#set bmargin 0
set yrange [400:2400]
set ytic 400, 200, 2400
set ylabel "Decay Heat (Watt/Assembly)"
#set format y "10^{%L}"
#set log y
set ytics nomirror
set mytics 5 

# set lines up
set style line 1 lt 2 lw 1 pt 4 lc rgb "sea-green"
set style line 2 lt 1 lw 1 lc rgb "red"
set style line 3 lt 1 lw 3 lc rgb "blue"

#set style line 1 lt 1 lw 2 lc rgb "red"
#set style line 2 lt 1 lw 2 lc rgb "green"
#set style line 3 lt 1 lw 2 lc rgb "blue"
set style line 4 lt 1 lw 3 lc rgb "brown"
set style line 5 lt 1 lw 3 lc rgb "red"
set style line 6 lt 1 lw 3 lc rgb "purple"
set style line 7 lt 1 lw 3 lc rgb "black"
set style increment user
#set multiplot

#set origin 0,0.3
#set xtics nomirror (2, 4, 8, 16, 32)
#set xtics nomirror (16875, 38400, 86400, 118800, 307200)
set xrange [0:35]
set xlabel "Decay Time (Years)
set mxtics 5
set xtics nomirror
#turn off the second axes
# move legend
#set key upper right Right

theta(t) = a + b * t**c + d * exp(-t)
fit theta(x) 'fig_5_1.data' using 1:2 via a, b, c, d
#theta(t) = c + a * exp(-b * t) + d * exp(-e * t)
#fit theta(x) 'fig_5_1.data' using 1:2 via a, b, c, d, e

plot 'fig_5_1.data' with points ls 1 title "DOE - Decay Heat History", \
      theta(x) ls 2 title "Best Fit Curve"

set term pop
replot
reset
