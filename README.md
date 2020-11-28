# SMPR

## get start

conda install pytorch=1.4.0 cudatoolkit=10.1 torchvision=0.5.0

pip install cython

******************

git clone https://github.com/cocodataset/cocoapi.git

cd coco/PythonAPI

python setup.py build_ext --inplace

python setup.py build_ext install

******************

pip install Pillow==6.2.2

pip install -v -e .
