
# ./darknet detector sequential ./cfg/coco.data ./cfg/yolov4-tiny.cfg ./weights/yolov4-tiny.weights data/dog.jpg -core_id 3 -num_exp 30

# ./darknet detector data-parallel ./cfg/coco.data ./cfg/yolov4-tiny.cfg ./weights/yolov4-tiny.weights data/dog.jpg -num_thread 3 -num_exp 30

# ./darknet detector data-parallel-mp ./cfg/coco.data ./cfg/yolov4-tiny.cfg ./weights/yolov4-tiny.weights data/dog.jpg -num_process 3 -num_exp 30 

# ./darknet detector gpu-accel ./cfg/coco.data ./cfg/yolov4-tiny.cfg ./weights/yolov4-tiny.weights data/dog.jpg -num_thread 3 -glayer 10 -num_exp 30

# ./darknet detector gpu-accel-mp ./cfg/coco.data ./cfg/yolov4-tiny.cfg ./weights/yolov4-tiny.weights data/dog.jpg -num_process 3 -glayer 10 -num_exp 30

./darknet detector cpu-reclaiming ./cfg/coco.data ./cfg/yolov4-tiny.cfg ./weights/yolov4-tiny.weights data/dog.jpg -num_thread 3 -glayer 10 -rlayer 20 -num_exp 50

# ./darknet detector cpu-reclaiming-mp ./cfg/coco.data ./cfg/yolov4-tiny.cfg ./weights/yolov4-tiny.weights data/dog.jpg -num_process 3 -glayer 15 -rlayer 20 -num_exp 30

