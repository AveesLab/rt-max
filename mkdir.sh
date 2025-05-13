mkdir measure

get_model_info() {
    case "$1" in
        "densenet201")
            data_file="imagenet1k"
            layer_num=306
            ;;
        "resnet152")
            data_file="imagenet1k"
            layer_num=206
            ;;
        "enetb0")
            data_file="imagenet1k"
            layer_num=136
            ;;
        "csmobilenet-v2")
            data_file="imagenet1k"
            layer_num=81
            ;;
        "squeezenet")
            data_file="imagenet1k"
            layer_num=50
            ;;
        "yolov7")
            data_file="coco"
            layer_num=143
            ;;
        "yolov7-tiny")
            data_file="coco"
            layer_num=99
            ;;
        "yolov4")
            data_file="coco"
            layer_num=162
            ;;
        "yolov4-tiny")
            data_file="coco"
            layer_num=38
            ;;
        *)
            echo "Unknown model: $1"
            exit 1
            ;;
    esac
}

## Layer time
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p measure/layer_time/$model/
done

## Sequential
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p measure/sequential/$model/
done

## Sequential-multiblas
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p measure/sequential-multiblas/$model/
done

## Pipeline
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p measure/pipeline/$model/
done


## Data-Parallel
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p measure/data-parallel/$model/
    for ((num_worker=1; num_worker<=11; num_worker++))
    do
        mkdir -p measure/data-parallel/$model/worker$num_worker/
    done
done

## CPU/GPU Layer Time
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p measure/layer_time/$model/
    for ((num_worker=1; num_worker<=11; num_worker++))
    do
        mkdir -p measure/layer_time/$model/worker$num_worker/
    done
done

## GPU-Accel
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
    get_model_info "$model"  # 모델에 맞는 layer_num 설정
    mkdir -p measure/gpu-accel/$model/
    
    for ((num_worker=1; num_worker<=11; num_worker++))
    do
        for ((Gstart=0; Gstart<=layer_num; Gstart++))
        do
            mkdir -p measure/gpu-accel/$model/gpu_task_log/worker$num_worker/G$Gstart/
            mkdir -p measure/gpu-accel/$model/worker_task_log/worker$num_worker/G$Gstart/
        done
    done
done