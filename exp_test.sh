#bin/bash

num_exp=0
num_start=10
num_end=50
num_exe=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
for num_exp in {10..100..10}; do
	./darknet detector gpu-accel_glayer ./cfg/coco.data ./cfg/yolov7-tiny.cfg ./weights/yolov7-tiny.weights data/dog.jpg -num_thread 8 -glayer 80 -num_exp $num_exp
done
