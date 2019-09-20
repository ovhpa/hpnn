#!/bin/bash
# Generate a kernel ANN filled with random weight.
# hubert.valencia@imass.nagoya-u.ac.jp  -- [OVHPA]

# gen.bash num_input num_output num_hidden

if [ $# -lt 3 ]; then
	echo "usage: gen.bash num_input num_hid1_out ... num_hidN_out num_output"
	echo "num_input: number of inputs"
	echo "num_hid1_out: number of outputs for hidden layer 1"
	echo "..."
	echo "num_hidN_out: number of outputs for hidden layer N"
	echo "num_output: number of outputs"
	exit 1
fi
N_INPUT=$1
if [ $N_INPUT -lt 1 ]; then
	echo "ERROR: number of inputs < 1"
	exit 1
fi
echo "[name] auto"
echo "[param] $@"
IDX=0
for param in $@
do
	rm -f random.tmp
	if [ $IDX -lt 1 ];then
		echo "[input] $param"
		IDX=1
		PREV=$param
		continue
	fi
	if [ $((IDX + 1)) -eq $# ];then
		echo "[output] $param"
		ALL_JDX=`seq $param`
		for jdx in ${ALL_JDX}
		do
			echo "[neuron $jdx] $PREV"
			TOTAL_BYTE=`echo "$PREV * 2" | bc -l`
			dd if=/dev/urandom of=random.tmp bs=$TOTAL_BYTE count=1 &> /dev/null
			WEIGHT=`hexdump -d random.tmp | sed -e 's/^[0-9a-f]*\ \(.*\)/\1/g' | sed -e 's/\ \([0-9]\)/0.\1/g' | grep "0\." | tr -d "\n"`
			awk -v var="$param $WEIGHT" 'BEGIN{split(var,list," "); for (i=2;i<=length(list);i++) printf "%7.5f ",2.0*(list[i]-0.5)/sqrt(list[1])}'
			echo
			rm -f random.tmp
		done
		break
	fi
	echo "[hidden $IDX] $param"
	ALL_JDX=`seq $param`
	for jdx in ${ALL_JDX}
	do
		echo "[neuron $jdx] $PREV"
		TOTAL_BYTE=`echo "$PREV * 2" | bc -l`
		dd if=/dev/urandom of=random.tmp bs=$TOTAL_BYTE count=1 &> /dev/null
		WEIGHT=`hexdump -d random.tmp | sed -e 's/^[0-9a-f]*\ \(.*\)/\1/g' | sed -e 's/\ \([0-9]\)/0.\1/g' | grep "0\." | tr -d "\n"`
		awk -v var="$param $WEIGHT" 'BEGIN{split(var,list," "); for (i=2;i<=length(list);i++) printf "%7.5f ",2.0*(list[i]-0.5)/sqrt(list[1])}'
		echo
		rm -f random.tmp
	done
	PREV=$param
	IDX=`echo "$IDX + 1" | bc -l`
done

