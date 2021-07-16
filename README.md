## Advancecd Deep Learning for Computer Vision

Project for ADL4CV: Shape Completion with Meso-Skeleton Learning by Dongyue Lu and Zhisheng Zheng, supervised by Yinyu Nie.
This code is tested under Python 3.6.3, PyTorch 1.2.0 on Ubuntu 18.04 and 20.04.

### 1. Installation

Simply run the following commands.

First, you need to set up pointnet++ dependencies
```shell
cd pointnet2
python setup.py install
```

Then, install `knn_cuda` by running the following command

```
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```


### 2. Data Preparation

Download training data: [airplane](https://drive.google.com/file/d/1OFD37G6jLN7gP36oRJpwkIF-iaQMdsf-/view?usp=sharing), [chair](https://drive.google.com/file/d/1ILLPJ2af55H1-ZSgWwyvLOsBuEaNxFr-/view?usp=sharing), [chair-200skeleton](https://drive.google.com/file/d/1wA7lz7iqIlY23Zmv6uEEMDOZg2Eci7zW/view?usp=sharing), [chair-400skeleton](https://drive.google.com/file/d/1gOfVmcByHH9GWE-lLTQkT57Pk8IQg5lq/view?usp=sharing), [chair-1200skeleton](https://drive.google.com/file/d/10dOplN5NizplW6XJ3rrA9N0NAWeZL-h5/view?usp=sharing).

Put the data into folder `datas`.


### 3. Train

Simply run the following code

```shell
python train_PFnet.py
```
to run the PFnet-based netwotk, or run
```shell
python train_PUnet.py
```
to run the PUnet-based network
See python `train.py --help` for all the training options. 


### 4. Test

Simply run the following code

```shell
python test_PFnet.py
```
or
```shell
python test_PUnet.py
```
See python `test.py --help` for all the testing options. 


### Reference
[PU-Net](https://github.com/lyqun/PU-Net_pytorch)

[PF-Net](https://github.com/zztianzz/PF-Net-Point-Fractal-Network)

[Point2Skeleton](https://github.com/clinplayer/Point2Skeleton)

[1]Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 652–660, 2017. 1

[2] Charles R Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. arXiv preprint arXiv:1706.02413, 2017. 1

[3] Angela Dai, Charles Ruizhongtai Qi, and Matthias Nießner. Shape completion using 3d-encoder-predictor cnns and shape synthesis. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5868–5877,2017. 1

[4] Yinyu Nie, Yiqun Lin, Xiaoguang Han, Shihui Guo, Jian Chang, Shuguang Cui, and Jian Jun Zhang. Skeleton-bridged point completion: From global inference to local adjustment. arXiv preprint arXiv:2010.07428, 2020. 1

[5] Lequan Yu, Xianzhi Li, Chi-Wing Fu, Daniel Cohen-Or, and Pheng-Ann Heng. Pu-net: Point cloud upsampling network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2790–2799, 2018. 1, 2

[6] Zitian Huang, Yikuan Yu, Jiawen Xu, Feng Ni, and Xinyi Le. Pf-net: Point fractal network for 3d point cloud completion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7662–7670, 2020. 2

[7] Cheng Lin, Changjian Li, Yuan Liu, Nenglun Chen, Yi-King Choi, and Wenping Wang. Point2skeleton: Learning skeletal representations from point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4277–4286, 2021. 2

[8] Angel X. Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, Jianxiong Xiao, Li Yi, and Fisher Yu. Shapenet: An information-rich 3d model repository, 2015. 2