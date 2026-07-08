#!/bin/bash

# Plot the output of `./k-means XX demo.in`; this script is run by
# "make demo". It expects that there are files `clusters_*.txt` and
# `centroids_*.txt` containing the coordinates of data points and
# centroids at each step, both of dimensionality 2.
#
# Last modified on 2025-12-03 by Moreno Marzolla.

for CLU in clusters_*.txt; do
    # Remove from $CLU every character which is NOT a digit
    STEP=`echo $CLU | tr -c -d [0-9]`
    CEN="centroids_${STEP}.txt"
    IMG="img_${STEP}.png"
    cat <<EOF | gnuplot
set term png linewidth 1.5 size 1024,768
set output "$IMG"
unset colorbox
unset xtics
unset ytics
set title "Step $STEP"
plot "$CLU" using 1:2:3 with p pt 6 palette notitle, \
     "$CEN" using 1:2 with p pt 2 ps 4 lt -1 lw 3 notitle
EOF
done
