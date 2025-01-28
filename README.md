# MM-GTUNets
This repository is the official PyTorch implementation of "MM-GTUNets: Unified Multi-Modal Graph Deep
Learning for Brain Disorders Prediction".

![MM-GTUNets](./MM_GTUNets.png)

## Contents
0. [Installation](#installation)
0. [Configuration](#configurationn)
0. [Data](#data)
0. [Training](#train)
0. [Testing](#testing)

## Installation
Code developed and tested in Python 3.9.0 using PyTorch 2.0.0. Please refer to their official websites for installation and setup.

Some major requirements are given below:

```python
numpy~=1.26.2
scikit-learn~=1.2.2
scipy~=1.10.1
torch~=2.0.0
torch-cluster~=1.6.0
torch-geometric~=2.0.4
torch-scatter~=2.0.9
torch-sparse~=0.6.13
torch-spline-conv~=1.2.1
nilearn~=0.10.1
```

Alternatively, you can choose to run the following code to install the required environment:
```shell
pip install -r requirements.txt
```

## Configuration

Please see Configs >>>[here](./opt.py/) Lines 44-96<<<.

| Configs    | Custom key                          | note              |
|------------|-------------------------------------|-------------------|
| dataset    | ABIDE/ADHD-200                      | dataset           |
| seed       | random seeds                        | default is 911    |
| train      | train or test (int)                 | 1-train, 0-test   |
| fold       | 10 (int)                            | k-fold validation |
| early_stop | early stop patience (int)           | default is 100    |
| lr         | initial model learning rate (float) | default is 1e-4   |
| vae_lr     | initial vae learning rate (float)   | default is 1e-3   |
| epoch      | number of epochs for training (int) | default is 500    |

other args:
* `--node_dim` dimension of node features after modality alignment
* `--img_depth` depth of the img_unet
* `--ph_depth` depth of the ph_unet
* `--hidden` hidden channels of the unet
* `--out` out channels of the unet
* `--dropout` ratio of dropout
* `--edge_drop` ratio of edge dropout
* `--pool_ratios` pooling ratio to be used in the Graph_Unet
* `--smh` graph_loss_smooth
* `--deg` graph_loss_degree
* `--val` graph_loss_value


## Data
### ABIDE
To fetch ABIDE public datasets.
```shell
python fetch_abide.py
```

### ADHD-200
The pre-processed ADHD-200 data upload address is as follows:

#### Google Drive

Link：https://drive.google.com/drive/folders/19HoajzuBFIV0dVGLtWv_jx2c0qg9srX_?usp=sharing 


#### Baidu Cloud Drive

Link：https://pan.baidu.com/s/16sqz0fZvuSHHypMkLtikbA 

Password：qj12

## Training

Classification Task ( Default dataset is ABIDE )
```shell
python train_mm_gtunets.py --train 1
```

## Testing

Classification Task ( Default dataset is ABIDE ) 
```shell
python train_mm_gtunets.py --train 0
```
## Citation
If you find our codes helpful, please star our project and cite our following papers: 

```
@article{cai2024mm,
  title={MM-GTUNets: Unified Multi-Modal Graph Deep Learning for Brain Disorders Prediction},
  author={Cai, Luhui and Zeng, Weiming and Chen, Hongyu and Zhang, Hua and Li, Yueyang and Yan, Hongjie and Bian, Lingbin and Wang, Nizhuan},
  journal={arXiv preprint arXiv:2406.14455},
  year={2024}
}
```
