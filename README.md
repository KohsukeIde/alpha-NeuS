# alpha-NeuS

This paper introduces α-NeuS, a new method for simultaneously reconstructing thin transparent objects and opaque objects based on neural implicit surfaces (NeuS). 

## [Paper](https://arxiv.org/abs/2106.10689) | [Data](https://www.dropbox.com/scl/fi/q8by01z58c0c6ioba4zq5/data.zip?rlkey=t29d79z51c679ztjvspd8t0pf&st=88igep20&dl=0)
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
