#!/bin/bash

lastinit=$(ls -t | grep log_init | head -n 1)
# n=4000
n=1053
if [[ ! -z $lastinit ]]; then
    gnuplot <<- EOF
        set term png
        set output "convergence-init.png"
        set xlabel "Iterations"
        set ylabel "Loss"
        unset key
        plot "$lastinit" using 1:2 every $n with lines
        set term wxt
EOF
fi

laste2e=$(ls -t | grep log_e2e | head -n 1)
n=400
if [[ ! -z $laste2e ]]; then
    gnuplot <<- EOF
        set term png
        set output "convergence-e2e.png"
        set xlabel "Iterations"
        set ylabel "Loss"
        unset key
        plot "$laste2e" using 1:2 every $n with lines
        set term wxt
EOF
fi
