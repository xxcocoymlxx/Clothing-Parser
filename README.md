### Clothing Parser

Mengning Yang

### Introduction

This is a model implemented using DeepLab-V3-Plus with Resnet Backbone to perform semantic segmentation on fashion images.

The data is adapted from the following paper:

Kota Yamaguchi, M Hadi Kiapour, Luis E Ortiz, Tamara L Berg, "Parsing Clothing in Fashion Photographs", CVPR 2012
http://vision.is.tohoku.ac.jp/~kyamagu/research/clothing_parsing/


### Dependency:
Environment: Python 3.6, Pytorch 1.0.1, CUDA 9.0, GTX 1060 6GB.
```Shell
conda install numpy
conda install tqdm
conda install pytorch torchvision -c pytorch
pip install torchsummary
conda install -c conda-forge tensorboardx
conda install scipy
conda install matplotlib
conda install -c conda-forge scikit-image
```

### Usage of predict.py:
```Shell
usage: predict.py -t clothes -i input_path -o output_path

```


### Acknowledgement
[pytorch-deeplab-xception]https://github.com/jfzhang95/pytorch-deeplab-xception


