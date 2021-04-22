#!/bin/bash

lastinit=$(ls -t | grep log_init | head -n 1)
[[ ! -z $lastinit ]] && gnuplot -persist <<-EOFMarker
    set term png
    set output "convergence-init.png"
    set xlabel "Iterations"
    set ylabel "Loss"
    unset key
    plot "$lastinit" using 1:2 every 4000 with lines
    set term wxt
EOFMarker

laste2e=$(ls -t | grep log_e2e | head -n 1)
[[ ! -z $laste2e ]] && gnuplot -persist <<-EOFMarker
    set term png
    set output "convergence-e2e.png"
    set xlabel "Iterations"
    set ylabel "Loss"
    unset key
    plot "$laste2e" using 1:2 every 400 with lines
    set term wxt
EOFMarker
