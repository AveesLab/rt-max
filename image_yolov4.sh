
# ./darknet detector sequential ./cfg/coco.data ./cfg/yolov4-tiny.cfg ./weights/yolov4-tiny.weights data/dog.jpg -core_id 3

# ./darknet detector data-parallel ./cfg/coco.data ./cfg/yolov4-tiny.cfg ./weights/yolov4-tiny.weights data/dog.jpg -num_thread 3

./darknet detector data-parallel-mp ./cfg/coco.data ./cfg/yolov4-tiny.cfg ./weights/yolov4-tiny.weights data/dog.jpg -num_thread 3

# ./darknet detector gpu-accel ./cfg/coco.data ./cfg/yolov4-tiny.cfg ./weights/yolov4-tiny.weights data/dog.jpg -num_thread 3

# ./darknet detector data-parallel ./cfg/coco.data ./cfg/yolov4-tiny.cfg ./weights/yolov4-tiny.weights data/dog.jpg -num_thread 3

