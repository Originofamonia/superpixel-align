#!/bin/bash

N_JOBS=25
N_DATA=2975

step=$(expr ${N_DATA} / ${N_JOBS} + 1)
gpu_i=0
i=0
while [ $i -lt $N_DATA ]
do
    start_i=$i
    i=$(expr $i + $step)
    if [ $i -ge $N_DATA ]
    then
        i=$N_DATA
    fi
    echo "START:$start_i END:$i"
    dmux run -- bash utils/dmux/create_labels_on_random300.sh $start_i $i
done

