#!/bin/bash
# Script to generate plots and downloading.
# Usage: $ ./fetch.sh

echo "Generating plots ..."
gcloud compute ssh n1s8-2nvtk80 --command "cd dsacstar && ./plot.sh"
echo "Downloading ..."
gcloud compute scp n1s8-2nvtk80:~/dsacstar/*.png .
