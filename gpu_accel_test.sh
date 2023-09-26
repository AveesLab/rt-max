
for var in {1..37}
	do
    ./darknet detector gpu-accel-mp ./cfg/coco.data ./cfg/yolov4-tiny.cfg ./weights/yolov4-tiny.weights data/dog.jpg -num_process 11 -glayer var -num_exp 10
	done


for var in {1..305}
	do
    ./darknet detector gpu-accel-mp ./cfg/imagenet1k.data ./cfg/densenet201.cfg ./weights/densenet201.weights data/dog.jpg -num_process 11 -glayer var -num_exp 10
	done