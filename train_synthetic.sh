#!/bin/bash

# train neus
python exp_runner.py --mode train --conf ./confs/womask_base.conf --case synthetic/snowglobe --gpu 0
python exp_runner.py --mode train --conf ./confs/womask_base.conf --case synthetic/case_double --gpu 0
python exp_runner.py --mode train --conf ./confs/womask_base.conf --case synthetic/jar --gpu 0
python exp_runner.py --mode train --conf ./confs/womask_base.conf --case synthetic/jug --gpu 0
python exp_runner.py --mode train --conf ./confs/womask_base.conf --case synthetic/bottle --gpu 0

# validate neus (iso<=0)
python exp_runner.py --is_continue --mode validate_mesh --conf ./confs/womask_base.conf --case synthetic/snowglobe --gpu 0 --mcube_threshold -0.0
python exp_runner.py --is_continue --mode validate_mesh --conf ./confs/womask_base.conf --case synthetic/case_double --gpu 0 --mcube_threshold -0.0
python exp_runner.py --is_continue --mode validate_mesh --conf ./confs/womask_base.conf --case synthetic/jar --gpu 0 --mcube_threshold -0.0
python exp_runner.py --is_continue --mode validate_mesh --conf ./confs/womask_base.conf --case synthetic/jug --gpu 0 --mcube_threshold -0.0
python exp_runner.py --is_continue --mode validate_mesh --conf ./confs/womask_base.conf --case synthetic/bottle --gpu 0 --mcube_threshold -0.0

# validate dcudf
python exp_runner.py --is_continue --mode validate_dcudf --conf ./confs/womask_base.conf --case synthetic/snowglobe --gpu 0 --mcube_threshold 0.005
python exp_runner.py --is_continue --mode validate_dcudf --conf ./confs/womask_base.conf --case synthetic/case_double --gpu 0 --mcube_threshold 0.005
python exp_runner.py --is_continue --mode validate_dcudf --conf ./confs/womask_base.conf --case synthetic/jar --gpu 0 --mcube_threshold 0.005
python exp_runner.py --is_continue --mode validate_dcudf --conf ./confs/womask_base.conf --case synthetic/jug --gpu 0 --mcube_threshold 0.005
python exp_runner.py --is_continue --mode validate_dcudf --conf ./confs/womask_base.conf --case synthetic/bottle --gpu 0 --mcube_threshold 0.005



