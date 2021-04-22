#!/bin/bash

lastinit=$(ls -t | grep log_init | head -n 1)
if [[ ! -z $lastinit ]]; then
    echo "here $lastinit"
    gnuplot <<- EOF
        set term png
        set output "convergence-init.png"
        set xlabel "Iterations"
        set ylabel "Loss"
        unset key
        plot "$lastinit" using 1:2 every 4000 with lines
        set term wxt
EOF
fi

laste2e=$(ls -t | grep log_e2e | head -n 1)
if [[ ! -z $laste2e ]]; then
    echo "here2"
    gnuplot <<- EOF
        set term png
        set output "convergence-e2e.png"
        set xlabel "Iterations"
        set ylabel "Loss"
        unset key
        plot "$laste2e" using 1:2 every 400 with lines
        set term wxt
EOF
fi
