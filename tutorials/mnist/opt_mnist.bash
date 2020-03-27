#!/bin/bash
# prepare monitor
rm -f tmp.mon
cat >tmp.mon <<!
#!/bin/bash
IDX=\`wc -l < raw\`
if [ "\$IDX" -gt 1 ]; then
  ./plot.gnuplot
fi
let "IDX += 1"
sed -e 's/\([0-9]\+\)\ *\([0-9]*\.[0-9]\)\ *\([0-9]*\.[0-9]\)$/ITER[\1] PASS = \2% OPT = \3%/g' raw
NTR=\`grep TRAINING ./log | wc -l\`
XTR=\`echo "scale=1;100*\$NTR/60000" | bc -l\`
XOP=\`echo "scale=0;-1 + 10*\$NTR/60000" | bc -l\`
if [ "\$XOP" -lt 0 ]; then
        MOP=".........."
else
        MOP=\`seq 0 9 | sed -e 's/[0-'\$XOP']/#/g' | sed -e 's/[0-9]/\./g' | tr -d '\n'\`
fi
# display
echo "ITER[\$IDX] [\$MOP](\$XTR%)"
!
chmod +x ./tmp.mon
#launch monitor
rm -f raw
touch raw
rm -f log
touch log
watch -t -n5 ./tmp.mon &
WPID=$!
# first pass
./train_nn -v -v -O4 -B4 ./mnist_snn.conf &> log
./run_nn -v -v -v -v -O4 -B4 ./cont_mnist_snn.conf &> results
NRS=`grep PASS results | wc -l`
XRS=`echo "scale=1;100*$NRS/10000" |bc -l`
NOK=`grep OK ./log | wc -l`
XOK=`echo "scale=1;100*$NOK/60000" |bc -l`
echo "1 $XRS $XOK" > raw
for IDX in `seq 2 30`
do
  ./train_nn -v -v -O4 -B4 ./cont_mnist_snn.conf &> log
  ./run_nn -v -v -v -v -O4 -B4 ./cont_mnist_snn.conf &> results
  NRS=`grep PASS results | wc -l`
  XRS=`echo "scale=1;100*$NRS/10000" |bc -l`
  NOK=`grep OK ./log | wc -l`
  XOK=`echo "scale=1;100*$NOK/60000" |bc -l`
  echo "$IDX $XRS $XOK" >> raw
done
sleep 6
kill $WPID
echo "All DONE!"
