#!/bin/bash

lastinit=$(ls -t | grep log_init_iter | head -n 1)
if [[ ! -z $lastinit ]]; then
    gnuplot <<- EOF
        set term png
        set output "conv-init-iter.png"
        set xlabel "Iterations"
        set ylabel "Loss"
        unset key
        plot "$lastinit" using 1:2 with lines
        set term wxt
EOF
    gnuplot <<- EOF
        set term png
        set output "conv-init-iter-pvsc.png"
        set xlabel "Iterations"
        set ylabel "% of Valid Scene Cooridnates"
        unset key
        plot "$lastinit" using 1:3
        set term wxt
EOF
fi

lastinit=$(ls -t | grep log_init_epoch | head -n 1)
if [[ ! -z $lastinit ]]; then
    gnuplot <<- EOF
        set term png
        set output "conv-init-epoch.png"
        set xlabel "Epochs"
        set ylabel "Mean Loss"
        unset key
        plot "$lastinit" using 1:2 with lines
        set term wxt
EOF
    gnuplot <<- EOF
        set term png
        set output "conv-init-epoch-pvsc.png"
        set xlabel "Epochs"
        set ylabel "% of Valid Scene Cooridnates"
        unset key
        plot "$lastinit" using 1:3 with lines
        set term wxt
EOF
fi

laste2e=$(ls -t | grep log_e2e_iter | head -n 1)
if [[ ! -z $laste2e ]]; then
    gnuplot <<- EOF
        set term png
        set output "conv-e2e-iter.png"
        set xlabel "Iterations"
        set ylabel "Loss"
        unset key
        plot "$laste2e" using 1:2 with lines
        set term wxt
EOF
fi

laste2e=$(ls -t | grep log_e2e_epoch | head -n 1)
if [[ ! -z $laste2e ]]; then
    gnuplot <<- EOF
        set term png
        set output "conv-e2e-epoch.png"
        set xlabel "Epochs"
        set ylabel "Mean Loss"
        unset key
        plot "$laste2e" using 1:2 with lines
        set term wxt
EOF
fi
