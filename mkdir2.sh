num_thread=6
mkdir measure
cd measure

models=(
	"yolov7-tiny" "densenet201" "resnet152" "enetb0"
	#"yolov4" "yolov4-tiny" "yolov7" "csmobilenet-v2" "squeezenet"
	)
archs=(
	"sequential" "pipeline" "data-parallel" "data-parallel_nano" "gpu-accel" "gpu-accel-nano"
    # "data-parallel_jitter" "data-parallel_r_test" "data-parallel_sleep" "data-parallel-mp" "data-parallel_r_test_jitter"
    # "gpu-accel_1thread" "gpu-accel_jitter" "gpu-accel-mp-reverse" "gpu-accel-GC" "gpu-accel-mp" "gpu-accel-CG" "gpu-accel_layer_test"
    # "cpu-reclaiming-CRG" "cpu-reclaiming-mp" "cpu-reclaiming" "cpu-reclaiming-GRC" "cpu-reclaiming-RGC" "cpu-reclaiming-GCR" "cpu-reclaiming-RCG"
    # "layer_time" "sequential-multiblas"
)
gpu_accel_archs=(
	"gpu-accel" "gpu-accel-nano"
	# "gpu-accel_1thread" "gpu-accel_jitter" "gpu-accel-mp-reverse"
    # "gpu-accel-CG" "gpu-accel_layer_test"
    # "gpu-accel-GC" "gpu-accel-mp"
)

for arch in "${archs[@]}"
do
	for model in "${models[@]}"
	do
		mkdir -p $arch/$model/
	done
done

for thread in $(seq 1 $num_thread)
do
	echo $thread
	for gpu_accel_arch in "${gpu_accel_archs[@]}"
	do
		for model in "${models[@]}"
		do
			# echo $gpu_accel_arch/${model}-multithread/${thread}thread/${var}glayer
			mkdir -p $gpu_accel_arch/${model}-multithread/${thread}thread/
			
		done
	done
done


# ## GPU-Accel_GC
# for num_thread in $(seq 1 11);
# do
# 	mkdir -p gpu-accel-GC/densenet201-multithread/${num_thread}thread/
# done

# ## GPU-Accel_GC
# for num_thread in $(seq 1 11);
# do
# 	mkdir -p gpu-accel-GC/resnet152-multithread/${num_thread}thread/
# done

# ## GPU-Accel_GC
# for num_thread in $(seq 1 11);
# do
# 	mkdir -p gpu-accel-GC/csmobilenet-v2-multithread/${num_thread}thread/
# done

# ## GPU-Accel_GC
# for num_thread in $(seq 1 11);
# do
# 	mkdir -p gpu-accel-GC/enetb0-multithread/${num_thread}thread/
# done

# ## GPU-Accel_GC
# for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
# do
# 	mkdir -p gpu-accel-nano/$model/
# done

# ## GPU-Accel_GC
# for num_thread in $(seq 1 11);
# do
# 	mkdir -p gpu-accel-nano/densenet201-multithread/${num_thread}thread/
# done

# ## GPU-Accel_GC
# for num_thread in $(seq 1 11);
# do
# 	mkdir -p gpu-accel-nano/resnet152-multithread/${num_thread}thread/
# done

# ## GPU-Accel_GC
# for num_thread in $(seq 1 11);
# do
# 	mkdir -p gpu-accel-nano/csmobilenet-v2-multithread/${num_thread}thread/
# done

# ## GPU-Accel_GC
# for num_thread in $(seq 1 11);
# do
# 	mkdir -p gpu-accel-nano/enetb0-multithread/${num_thread}thread/
# done

# ## GPU-Accel_GC
# for num_thread in $(seq 1 11);
# do
# 	mkdir -p gpu-accel-CG/densenet201-multithread/${num_thread}thread/
# done
