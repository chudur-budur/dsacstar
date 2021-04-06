#!/usr/bin/gnuplot

set term png
set output "convergence.png"
set xlabel "Iterations"
set ylabel "Loss"
unset key
plot "log_init_7scenes_chess_rgb_06-04-21-02-48-14.txt" using 1:2 every 4000 with lines
set term wxt
