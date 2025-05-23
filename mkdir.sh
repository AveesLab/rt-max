mkdir measure

get_model_info() {
    case "$1" in
        "resnet10")
            data_file="imagenet1k"
            layer_num=10
            ;;
        "resnet18")
            data_file="imagenet1k"
            layer_num=18
            ;;
        "yolov2-tiny")
            data_file="coco"
            layer_num=9
            ;;
        "yolov4-tiny")
            data_file="coco"
            layer_num=21
            ;;
        *)
            echo "Unknown model: $1"
            exit 1
            ;;
    esac
}

## Data-Parallel
for model in "resnet10" "resnet18" "yolov2-tiny" "yolov4-tiny" 
do
	mkdir -p measure/data-parallel/$model/
    for ((num_worker=1; num_worker<=8; num_worker++))
    do
        mkdir -p measure/data-parallel/$model/worker$num_worker/
    done
done

## CPU Layer Time
for model in "resnet10" "resnet18" "yolov2-tiny" "yolov4-tiny" 
do
	mkdir -p measure/pseudo_layer_time/$model/
    for ((num_worker=1; num_worker<=8; num_worker++))
    do
        mkdir -p measure/pseudo_layer_time/$model/cpu/worker$num_worker/
        mkdir -p measure/gpu_segments/$model/
    done
done

## GPU Layer Time
for model in "resnet10" "resnet18" "yolov2-tiny" "yolov4-tiny" 
do
    get_model_info "$model"  # 모델에 맞는 layer_num 설정
    mkdir -p measure/pseudo_layer_time/$model/
    
    for ((num_worker=1; num_worker<=8; num_worker++))
    do
        mkdir -p measure/gpu_segments/$model/gpu_task_log/worker$num_worker/
        mkdir -p measure/gpu_segments/$model/worker_task_log/worker$num_worker/
        for ((Gstart=0; Gstart<layer_num; Gstart++))
        do
            mkdir -p measure/pseudo_layer_time/$model/gpu/worker$num_worker/G$Gstart/
        done
    done
done

## GPU-Accel
for model in "resnet10" "resnet18" "yolov2-tiny" "yolov4-tiny" 
do
    get_model_info "$model"  # 모델에 맞는 layer_num 설정
    mkdir -p measure/gpu-accel/$model/
    
    for ((num_worker=1; num_worker<=8; num_worker++))
    do
        for ((Gstart=0; Gstart<layer_num; Gstart++))
        do
            mkdir -p measure/gpu-accel/$model/gpu_task_log/worker$num_worker/G$Gstart/
            mkdir -p measure/gpu-accel/$model/worker_task_log/worker$num_worker/G$Gstart/
        done
    done
done