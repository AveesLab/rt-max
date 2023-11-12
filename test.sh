# ./gpu_accel_pre_gpu_test.sh -model densenet201
# ./gpu_accel_gpu_test.sh -model densenet201
# ./gpu_accel_test.sh -model densenet201


# sleep 3
# ./sequential_test.sh -model densenet201
# sleep 3
# ./pipeline_test.sh -model densenet201
# sleep 3

# sleep 3
# ./data_parallel_test.sh -model densenet201
# sleep 3
# ./data_parallel_jitter_test.sh -model densenet201
# sleep 3
# ./data_parallel_sleep_test.sh -model densenet201
# sleep 3


#./data_parallel_r_test_test.sh -model densenet201
#sleep 3
#./data_parallel_r_test_jitter_test.sh -model densenet201
#sleep 3


# ./layer_time_test.sh -model densenet201
./gpu_layer_test.sh -model densenet201
