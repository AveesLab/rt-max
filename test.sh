#!/bin/bash
model=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -model)
            model="$2"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

if [ -z "$model" ]; then
    echo "Error: -model parameter not specified"
    exit 1
fi


# ./sequential_test.sh -isGPU 0 -model $model 
# ./sequential_test.sh -isGPU 1 -model $model 
# ./sequential_jitter_test.sh -isGPU 0 -model $model
# ./sequential_jitter_test.sh -isGPU 1 -model $model

# ./pipeline_test.sh -isGPU 0 -model $model
# ./pipeline_test.sh -isGPU 1 -model $model
# ./pipeline_jitter_test.sh -isGPU 0 -model $model
# ./pipeline_jitter_test.sh -isGPU 1 -model $model

# ./data_parallel_sleep_test.sh -model $model
# ./data_parallel_jitter_test.sh -model $model

./data_parallel_test.sh -model $model

if [ "$model" == "densenet201" ]; then
    ./gpu_accel_gpu_test.sh -model $model -gap 5
elif [ "$model" == "yolov7-tiny" ]; then
    ./gpu_accel_gpu_test.sh -model $model -gap 2
else
    echo "Error: Unsupported model '$model'"
    exit 1
fi

#./gpu_accel_gpu_test.sh -model $model

# -----------------------------------------------------------
# ./gpu_accel_pre_gpu_test.sh -model densenet201
#./gpu_accel_sleep_test.sh -model densenet201
#./gpu_accel_test.sh -model densenet201

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
#./gpu_layer_test.sh -model densenet201
