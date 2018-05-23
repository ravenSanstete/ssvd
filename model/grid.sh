#!/bin/bash

for ((i=1; i <= 20; i++)); do
    for ((j = 1; j <= 20; j++)); do
	python quan_svd.py --u_num_state $i --v_num_state $j
    done
done
