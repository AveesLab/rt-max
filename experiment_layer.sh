
./darknet detector sequential ./cfg/imagenet1k.data ./cfg/densenet201.cfg ./weights/densenet201.weights data/dog.jpg -core_id 1 -num_exp 30

for var in {1..11}
	do
    ./darknet detector sequential-multiblas ./cfg/imagenet1k.data ./cfg/densenet201.cfg ./weights/densenet201.weights data/dog.jpg -num_blas $var -num_exp 30
	done

for var in {1..305}
	do
    ./darknet detector gpu-accel ./cfg/imagenet1k.data ./cfg/densenet201.cfg ./weights/densenet201.weights data/dog.jpg -num_thread 1 -glayer $var -num_exp 30
	done