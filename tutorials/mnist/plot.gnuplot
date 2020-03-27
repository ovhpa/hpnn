#!/usr/bin/gnuplot
set term dumb size 80,30 aspect 1
set tics out
set y2tics
set key below
plot "raw" u 1:2 w lp t "PASS" axis x1y1, "raw" u 1:3 w lp t "OPT" axis x1y2
