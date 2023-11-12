./gpu_accel_pre_gpu_test.sh -model yolov4
./gpu_accel_gpu_test.sh -model yolov4
./gpu_accel_test.sh -model yolov4


sleep 3
./sequential_test.sh -model yolov4
sleep 3
./pipeline_test.sh -model yolov4
sleep 3
./data_parallel_test.sh -model yolov4
sleep 3

./data_parallel_jitter_test.sh -model yolov4
sleep 3
./data_parallel_sleep_test.sh -model yolov4
sleep 3


# ./layer_time_test.sh -model densenet201
