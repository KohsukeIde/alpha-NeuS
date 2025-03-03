# alpha-NeuS

This paper introduces α-NeuS, a new method for simultaneously reconstructing thin transparent objects and opaque objects based on neural implicit surfaces (NeuS). 

## [Project Page](https://lcs.ios.ac.cn/~houf/pages/alphaneus/index.html) | [Paper](https://arxiv.org/abs/2411.05362) | [Data](https://www.dropbox.com/scl/fi/q8by01z58c0c6ioba4zq5/data.zip?rlkey=t29d79z51c679ztjvspd8t0pf&st=74ks83w8&dl=0)
This is the official repo for the implementation of **From Transparent to Opaque: Rethinking Neural Implicit Surfaces with α-NeuS**.

## Usage

#### Data Convention
The data is organized as follows:

```
<case_name>
|-- cameras_xxx.npz    # camera parameters
|-- image
    |-- 000.png        # target image for each view
    |-- 001.png
    ...
|-- mask
    |-- 000.png        # target mask each view (For unmasked setting, set all pixels as 255)
    |-- 001.png
    ...
```

Here the `cameras_xxx.npz` follows the data format in [IDR](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md), where `world_mat_xx` denotes the world to image projection matrix, and `scale_mat_xx` denotes the normalization matrix.

use the following link to download code to convert colmap to npz https://github.com/wutong16/Voxurf/issues/11#issuecomment-1494710756

#### Environment
Set up the environment as specified in NeuS.

```sh
bash wget "https://dl.dropboxusercontent.com/scl/fi/q8by01z58c0c6ioba4zq5/data.zip?rlkey=t29d79z51c679ztjvspd8t0pf" -O data.zip
unzip data.zip
```



## Training
For synthetic scene
```sh
bash train_synthetic.sh
```
For real-world scene
```sh
bash train_real.sh
```

Or, you can train it step by step as follows:
1. train NeuS
```sh
python exp_runner.py --mode train --conf ${config_name} --case ${data_dirname}
```
2. validate mesh of NeuS
```sh
python exp_runner.py --is_continue --mode validate_mesh --conf ${config_name} --case ${data_dirname} --mcube_threshold -0.0
```
3. validate mesh by using dcudf
```sh
python exp_runner.py --is_continue --mode validate_dcudf --conf ${config_name} --case ${data_dirname} --mcube_threshold 0.005
```




## Acknowledgements & Citation
This work is built upon the foundation of [NeuS](https://github.com/totoro97/NeuS) and [DoubleCoverUDF](https://github.com/jjjkkyz/DCUDF). We offer our most sincere thanks to their outstanding work.

If you find our work useful, please feel free to cite us.
```bibtex
@inproceedings{zhang2024from,
	title={From Transparent to Opaque: Rethinking Neural Implicit Surfaces with $\alpha$-NeuS},
	author={Zhang, Haoran and Deng, Junkai and Chen, Xuhui and Hou, Fei and Wang, Wencheng and Qin, Hong and Qian, Chen and He, Ying},
	booktitle={Proceedings of the Neural Information Processing Systems (NeurIPS)},
	year={2024}
}
```
