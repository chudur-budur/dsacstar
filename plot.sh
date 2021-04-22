#!/bin/bash

lastinit=$(ls -t | grep log_init | head -n 1)
if [ -n $lastinit ]; then
gnuplot -persist <<-EOFMarker
    set term png
    set output "convergence-init.png"
    set xlabel "Iterations"
    set ylabel "Loss"
    unset key
    plot "$lastinit" using 1:2 every 4000 with lines
    set term wxt
EOFMarker
fi

laste2e=$(ls -t | grep log_e2e | head -n 1)
if [ -n $laste2e ]; then
gnuplot -persist <<-EOFMarker
    set term png
    set output "convergence-e2e.png"
    set xlabel "Iterations"
    set ylabel "Loss"
    unset key
    plot "$laste2e" using 1:2 every 400 with lines
    set term wxt
EOFMarker
fi
