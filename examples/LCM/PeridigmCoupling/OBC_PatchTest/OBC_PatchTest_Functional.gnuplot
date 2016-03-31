
set terminal pdf enhanced font "Times-Roman,24" size 12in, 8in 
set output "functional_different_num_proc.pdf"
set xlabel "ROL Iteration" font "Times-Roman,32"
set ylabel "Functional" font "Times-Roman,32"
#set xrange [-0.9:0.2]
#set yrange [-0.0006:0.0006]
set logscale y
set key top right
plot "functional_np1.txt" using 1:2 with points pt 7 ps 4 title "1 Processor ", \
     "functional_np2.txt" using 1:2 with points pt 7 ps 4 title "2 Processors", \
     "functional_np3.txt" using 1:2 with points pt 7 ps 4 title "3 Processors", \
     "functional_np4.txt" using 1:2 with points pt 7 ps 4 title "4 Processors", \
     "functional_np5.txt" using 1:2 with points pt 7 ps 4 title "5 Processors"
