#!/bin/bash

rs=4
dt=0.20
nw=512

python3 step1_opt.py --rs $rs

python3 step2_dmc.py --rs $rs --dt $dt --nconf $nw

fh5=rs${rs}-dt${dt}-n${nw}-dmc.h5

nequil=50
python3 dspork.py $fh5 -c energytotal -e $nequil -t

