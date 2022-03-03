The detection training demo:
* dataset: AIR-SARShip-1.0 
* target: ship
* model: faster_rcnn


Run the demo:

1. Install PaddleRS
```
git clone https://github.com/PaddleCV-SIG/PaddleRS.git
cd PaddleRS
pip install -r requirements.txt
python setup.py install
```

2. Run the demo
```
cd tutorials/train/detection/

# run training on single GPU
export CUDA_VISIBLE_DEVICES=0
python faster_rcnn_sar_ship.py

# run traing on multi gpu
export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch faster_rcnn_sar_ship.py
```
