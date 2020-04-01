#!/bin/bash
# tutorial demonstrating SNN with MNIST database
# hubert.valencia _at_ imass.nagoya-u.ac.jp    -- [OVHPA]

# modify to suits your train_nn best parameters
export OMP_NUM_THREADS=4
FIRST_TRAIN_ARG="-v -v -v -O4 -B4 ./mnist_ann.conf"
TRAIN_ARG="-v -v -v -O4 -B4 ./cont_mnist_ann.conf"
RUN_ARG="-v -v -O4 -B4 ./cont_mnist_ann.conf"
PMNIST_ARG="samples tests"

#check pmnist
if [ -x "./pmnist" ]&>/dev/null; then
        PMNIST=`pwd`/pmnist
else
        PMNIST=`which pmnist 2>/dev/null`
        if [ -z "$PMNIST" ]&>/dev/null; then
                echo "Can't find pmnist!"
                echo "Please make pmnist executable before executing tutorial!"
                exit
        fi
fi
#check train_nn
if [ -x "../../tests/train_nn" ]&>/dev/null; then
        TRAIN=`pwd`/../../tests/train_nn
else
        TRAIN=`which train_nn 2>/dev/null`
        if [ -z "$TRAIN" ]&>/dev/null; then
                echo "Can't train_nn!"
                echo "Please make train_nn before executing tutorial!"
                exit
        fi
fi
#check run_nn
if [ -x "../../tests/run_nn" ]&>/dev/null; then
        RUN=`pwd`/../../tests/run_nn
else
        RUN=`which run_nn 2>/dev/null`
        if [ -z "$RUN" ]&>/dev/null; then
                echo "Can't find run_nn!"
                echo "Please make run_nn before executing tutorial!"
                exit
        fi
fi
FIRST_TRAIN_CMD="$TRAIN $FIRST_TRAIN_ARG"
TRAIN_CMD="$TRAIN $TRAIN_ARG"
RUN_CMD="$RUN $RUN_ARG"
PMNIST_CMD="$PMNIST $PMNIST_ARG"
echo "For the tutorial,  the MNIST database is required"
echo "in the ./mnist directory. If you have not done so"
echo "already you can download the database zipped file"
echo "using this script... The size of download is 12MB"
echo "and after decompression,  an extra ~460MB will be"
echo "used on your hard drive."
while true; do
        read -n 1 -p "Download MNIST database? Y/N " answer
        case $answer in
        [Yy]* ) NEED_MNIST=y; break;;
        [Nn]* ) NEED_MNIST=; break;;
        * ) echo " Please answer Y/N.";;
        esac
done
echo
#download?
if [ -n "$NEED_MNIST" ]&>/dev/null; then
        if [ -d "./mnist" ]&>/dev/null; then
                echo "DIRECTORY ./mnist ALREADY EXISTS!"
                echo "Please remove it -can't continue!"
                exit
        fi
        echo "Downloading MNIST database..."
	mkdir -p mnist
	cd ./mnist
        mkdir -p temp
        cd temp
	wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
	wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
	#unpack & rename
	gunzip train-labels-idx1-ubyte.gz
	mv train-labels-idx1-ubyte ../train_labels
	gunzip train-images-idx3-ubyte.gz
	mv train-images-idx3-ubyte ../train_images
	gunzip t10k-labels-idx1-ubyte.gz
	mv t10k-labels-idx1-ubyte ../test_labels
	gunzip t10k-images-idx3-ubyte.gz
	mv t10k-images-idx3-ubyte ../test_images
	#we will keep downloaded database in temp
	cd ..
	cd ..
        echo "mnist directory ready"
else
        echo "NO download will be performed!"
fi
if [ -d "./mnist" ]&>/dev/null; then
        echo "starting from mnist directory..."
else
        echo "mnist directory is not present!"
        echo "Something went wrong..."
        exit
fi
# prepare samples
echo "preparing samples"
if [ -d "./mnist/samples" ]&>/dev/null; then
        if [ -d "./mnist/old-samples" ]&>/dev/null; then
                rm -f ./mnist/old-samples/*
                rmdir ./mnist/old-samples
        fi
        mv ./mnist/samples ./mnist/old-samples
fi
mkdir -p ./mnist/samples
#prepare tests
if [ -d "./mnist/tests" ]&>/dev/null; then
        if [ -d "./mnist/old-tests" ]&>/dev/null; then
                rm -f ./mnist/old-tests/*
                rmdir ./mnist/old-tests
        fi
        mv ./mnist/tests ./mnist/old-tests
fi
mkdir -p ./mnist/tests
cd mnist
eval $PMNIST_CMD
rm -f ./s0/* ./s1/* ./s2/* ./s3/* ./s4/* ./s5/*
mkdir -p ./s0 ./s1 ./s2 ./s3 ./s4 ./s5
cp ./samples/s0* ./s0
cp ./samples/s60000.txt ./s0
cp ./samples/s1* ./s1
cp ./samples/s2* ./s2
cp ./samples/s3* ./s3
cp ./samples/s4* ./s4
cp ./samples/s5* ./s5
echo "preparing configuration files"
cat > mnist_ann.conf <<!
[name] MNIST
[type] ANN
[init] generate
[seed] 10958
[input] 784
[hidden] 300
[output] 10
[train] BPM
[sample_dir] ./samples
[test_dir] ./tests
!
sed -e 's/^\[init\].*/[init] kernel.opt/g' -e 's/^\[seed\].*/[seed] 1/g' mnist_ann.conf > cont_mnist_ann.conf
# prepare monitor
rm -f tmp.mon*
cat > tmp.mon0 <<! 
#!/bin/bash
IDX=\`wc -l < raw\`
if [ "\$IDX" -gt 1 ]; then
  ./tmp.gnuplot
fi
let "IDX += 1"
tail -20 raw | sed -e 's/\([0-9]\+\)\ *\([0-9]*\.[0-9]\)\ *\([0-9]*\.[0-9]\)$/ITER[\1] PASS = \2% OPT = \3%/g'
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
chmod +x ./tmp.mon0
# prepare plots
cat > tmp.gnuplot <<!
#!/usr/bin/gnuplot
set term dumb size 80,30 aspect 1
set tics out 
set y2tics
set key below
plot "raw" u 1:2 w lp t "PASS" axis x1y1, "raw" u 1:3 w lp t "OPT" axis x1y2
!
chmod +x ./tmp.gnuplot
#launch monitor
rm -f raw 
touch raw 
rm -f log 
touch log 
watch -t -n5 ./tmp.mon0 &
WPID=$!
# first pass
eval $FIRST_TRAIN_CMD &> log
eval $RUN_CMD &> results
NRS=`grep PASS results | wc -l`
XRS=`echo "scale=1;100*$NRS/60000" |bc -l`
NOK=`grep OK ./log | wc -l`
XOK=`echo "scale=1;100*$NOK/60000" |bc -l`
echo "0 $XRS $XOK" > raw
ITER=1
kill $WPID
sed -e 's/60000/10000/g' tmp.mon0 > tmp.mon
chmod +x tmp.mon
rm -f log
touch log
watch -t -n5 ./tmp.mon &
WPID=$!
for IDX in `seq 2 50`
do
  for JDX in `seq 0 5`
  do
    sed -e 's/^\[init\].*/[init] kernel.opt/g' -e 's/^\[seed\].*/[seed] '$IDX'/g' -e 's/^\[sample_dir\].*/[sample_dir] .\/s'$JDX'/g' mnist_ann.conf > cont_mnist_ann.conf
    #do 5 time each
    for KDX in `seq 1 5`
    do
      eval $TRAIN_CMD &> log
      eval $RUN_CMD &> results
      NRS=`grep PASS results | wc -l`
      XRS=`echo "scale=1;100*$NRS/10000" |bc -l`
      NOK=`grep OK ./log | wc -l`
      XOK=`echo "scale=1;100*$NOK/10000" |bc -l`
      echo "$ITER $XRS $XOK" >> raw
      (( ITER += 1 ))
    done
  done
done
sleep 6
kill $WPID
echo "All DONE!"

