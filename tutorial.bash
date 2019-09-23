#!/bin/bash
# tutorial demonstrating ANN with RRUFF XRD database
# hubert.valencia@imass.nagoya-u.ac.jp    -- [OVHPA]

if [ -z "$OMP_NUM_THREADS" ]&>/dev/null; then
	export OMP_NUM_THREADS=4
fi

#check or make pdif
if [ -x "./pdif" ]&>/dev/null; then
	PDIF=./pdif
else
	make pdif
	if [ -x "./pdif" ]&>/dev/null; then
		PDIF=./pdif
	else
		echo "Can't make pdif executable!, bailing!"
		exit
	fi
fi
#check or make train_nn
if [ -x "./train_nn" ]&>/dev/null; then
	TRAIN=./train_nn
else
	make all
	if [ -x "./train_nn" ]&>/dev/null; then
		TRAIN=./train_nn
	else
		echo "Can't make train_nn executable!, bailing!"
		exit
	fi
fi
#prepare rruff
echo "For the tutorial,  the RRUFF database is required"
echo "in the ./rruff directory. If you have not done so"
echo "already you can download the database zipped file"
echo "using this script... The size of download is 95MB"
echo "and after decompression,  an extra ~360MB will be"
echo "used on your hard drive."
while true; do
	read -n 1 -p "Download RRUFF database? Y/N " answer
	case $answer in
	[Yy]* ) NEED_RRUFF=y; break;;
	[Nn]* ) NEED_RRUFF=; break;;
	* ) echo " Please answer Y/N.";;
	esac
done
echo

if [ -n "$NEED_RRUFF" ]&>/dev/null; then
	if [ -d "./rruff" ]&>/dev/null; then
		echo "DIRECTORY ./rruff ALREADY EXISTS!"
		echo "Please remove it -can't continue!"
		exit
	fi
	echo "Downloading RRUFF database..."
	mkdir -p temp
	cd temp
	wget http://rruff.info/zipped_data_files/powder/DIF.zip
	wget http://rruff.info/zipped_data_files/powder/XY_RAW.zip
	mkdir -p dif
	unzip DIF.zip -d ./dif/
	mkdir -p raw
	unzip XY_RAW.zip -d ./raw/
	echo "Re-naming the files in dif directory..."
	cd ./dif
	FLIST=`ls -1F`
	for curr_file in ${FLIST}
	do
		new_file=`echo $curr_file | sed -e 's/_/\ /g' | grep -o '[RX][0-9][0-9]*[^ ]*'`
		mv $curr_file $new_file.txt
	done
	echo "Re-naming the files in raw directory..."
	cd ../raw/
	FLIST=`ls -1F`
	for curr_file in ${FLIST}
	do
		new_file=`echo $curr_file | sed -e 's/_/\ /g' | grep -o '[RX][0-9][0-9]*[^ ]*'`
		mv $curr_file $new_file.txt
	done
	cd ../..
	mv temp rruff
	echo "rruff directory ready"
else
	echo "NO download will be performed!"
fi
if [ -d "./rruff" ]&>/dev/null; then
	echo "starting from rruff directory..."
else
	echo "rruff directory not present -bailing!"
	exit
fi
# prepare samples
if [ -d "./rruff/samples" ]&>/dev/null; then
	if [ -d "./rruff/old-samples" ]&>/dev/null; then
		rm -f ./rruff/old-samples/*
		rmdir ./rruff/old-samples
	fi
	mv ./rruff/samples ./rruff/old-samples
fi
mkdir -p ./rruff/samples
#prepare tests
if [ -d "./rruff/tests" ]&>/dev/null; then
        if [ -d "./rruff/old-tests" ]&>/dev/null; then
                rm -f ./rruff/old-tests/*
                rmdir ./rruff/old-tests
        fi
        mv ./rruff/tests ./rruff/old-tests
fi
mkdir -p ./rruff/tests
echo "preparing samples"
cd rruff
../pdif . -i 850 -o 230
echo "preparing configuration files"
echo "[name] tutorial" > nn.conf
echo "[type] ANN" >> nn.conf
echo "[init] generate" >> nn.conf
echo "[seed] 0" >> nn.conf
echo "[input] 851" >> nn.conf
echo "[hidden] 460" >> nn.conf
echo "[output] 230" >> nn.conf
echo "[train] BPM" >> nn.conf
echo "[sample_dir] ./samples" >> nn.conf
echo "[test_dir] ./tests" >> nn.conf
sed 's/generate/kernel.opt/g' nn.conf > nn2.conf
#all done
echo "training NN -- turn 0"
#first round
../train_nn
#next 10 rounds
for idx in `seq 10`
do
	echo "training NN -- turn $idx"
	../train_nn nn2.conf
done
echo "ANN should be trained enough for a rough test."
echo "Please try ../run_nn - from rruff directory..."








