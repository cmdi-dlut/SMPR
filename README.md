# SMPR

Code repository for the paper [SMPR: Single-Stage Multi-Person Pose Regression](https://arxiv.org/abs/2006.15576), by Junqi Lin, Huixin Miao, Junjie Cao, Zhixun Su and Risheng Liu.

## Getting Started

conda create -n mmdet python=3.7

conda activate mmdet

conda install pytorch=1.4.0 cudatoolkit=10.1 torchvision=0.5.0

pip install cython

git clone https://github.com/cocodataset/cocoapi.git

cd cocoapi/PythonAPI

python setup.py build_ext --inplace

python setup.py build_ext install

pip install -r requirements.txt

pip install -v -e .

## Pretrained Models

You can download the trained model on [Baidu Yun](https://pan.baidu.com/s/1S_7s_tfIHlqvCCKWFXyWGA)，with the extraction code：aaaa

## Inference

You can now evaluate the models on the COCO val2017 split:

```
./tools/dist_test.sh configs/SMPR/ResNet_50.py work_dirs/r50.pth 4 --eval keypoints --options "jsonfile_prefix=./work_dirs/r50"
```

## Acknowledgment

We would like to thank MMDetection team for producing this great object detection toolbox!

