#!/bin/bash

# train neus
python exp_runner.py --mode train --conf ./confs/womask_base.conf --case real/magician-box --gpu 0
python exp_runner.py --mode train --conf ./confs/womask_base.conf --case real/magician-plane --gpu 0
python exp_runner.py --mode train --conf ./confs/womask_base.conf --case real/sunglasses --gpu 0
python exp_runner.py --mode train --conf ./confs/womask_base.conf --case real/toy_box --gpu 0
python exp_runner.py --mode train --conf ./confs/womask_base.conf --case real/toy_cylinder --gpu 0

# validate neus (iso<=0)
python exp_runner.py --is_continue --mode validate_mesh --conf ./confs/womask_base.conf --case real/magician-box --gpu 0 --mcube_threshold -0.0
python exp_runner.py --is_continue --mode validate_mesh --conf ./confs/womask_base.conf --case real/magician-plane --gpu 0 --mcube_threshold -0.0
python exp_runner.py --is_continue --mode validate_mesh --conf ./confs/womask_base.conf --case real/sunglasses --gpu 0 --mcube_threshold -0.0
python exp_runner.py --is_continue --mode validate_mesh --conf ./confs/womask_base.conf --case real/toy_box --gpu 0 --mcube_threshold -0.0
python exp_runner.py --is_continue --mode validate_mesh --conf ./confs/womask_base.conf --case real/toy_cylinder --gpu 0 --mcube_threshold -0.0

# validate dcudf
python exp_runner.py --is_continue --mode validate_dcudf --conf ./confs/womask_base.conf --case real/magician-box --gpu 0 --mcube_threshold 0.005
python exp_runner.py --is_continue --mode validate_dcudf --conf ./confs/womask_base.conf --case real/magician-plane --gpu 0 --mcube_threshold 0.005
python exp_runner.py --is_continue --mode validate_dcudf --conf ./confs/womask_base.conf --case real/sunglasses --gpu 0 --mcube_threshold 0.005
python exp_runner.py --is_continue --mode validate_dcudf --conf ./confs/womask_base.conf --case real/toy_box --gpu 0 --mcube_threshold 0.005
python exp_runner.py --is_continue --mode validate_dcudf --conf ./confs/womask_base.conf --case real/toy_cylinder --gpu 0 --mcube_threshold 0.002



