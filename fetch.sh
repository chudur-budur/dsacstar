#!/bin/bash
# Script to generate plots and downloading.
# Usage: $ ./fetch.sh

probe=$(command -v gcloud)
if [ -n $probe ]; then
    echo "Generating plots ..."
    gcloud compute ssh dsacstar-jen7 --command "cd dsacstar && ./plot.sh"
    echo "Downloading ..."
    gcloud compute scp dsacstar-jen7:~/dsacstar/*.png .
fi

probe=$(command -v xdg-open)
if [ -n $probe ]; then
    xdg-open conv-init-epoch.png &
    xdg-open conv-init-epoch-pvsc.png &
    xdg-open conv-init-iter.png &
    xdg-open conv-init-iter-pvsc.png &
fi
