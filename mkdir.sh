mkdir measure
cd measure

## Layer time
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p layer_time/$model/
done

## Sequential
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p sequential/$model/
done

## Sequential-multiblas
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p sequential-multiblas/$model/
done

## Pipeline
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p pipeline/$model/
done


## Data-Parallel
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p data-parallel/$model/
done


## Data-Parallel-MP
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p data-parallel-mp/$model/
done

## data-parallel_r_test
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p data-parallel_r_test/$model/
done

## data-parallel_r_test_jitter
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p data-parallel_r_test_jitter/$model/
done

## data-parallel_sleep
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p data-parallel_sleep/$model/
done

## data-parallel_jitter
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p data-parallel_jitter/$model/
done

## data-parallel_nano
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p data-parallel_nano/$model/
done


## GPU-Accel
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p gpu-accel/$model/
done

## GPU-Accel_CG
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p gpu-accel-CG/$model/
done

## GPU-Accel_GC
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p gpu-accel-GC/$model/
done

## GPU-Accel_GC
for num_thread in $(seq 1 11);
do
	mkdir -p gpu-accel-GC/densenet201-multithread/${num_thread}thread/
done

## GPU-Accel_GC
for num_thread in $(seq 1 11);
do
	mkdir -p gpu-accel-GC/resnet152-multithread/${num_thread}thread/
done

## GPU-Accel_GC
for num_thread in $(seq 1 11);
do
	mkdir -p gpu-accel-GC/csmobilenet-v2-multithread/${num_thread}thread/
done

## GPU-Accel_GC
for num_thread in $(seq 1 11);
do
	mkdir -p gpu-accel-GC/enetb0-multithread/${num_thread}thread/
done

## GPU-Accel_GC
for num_thread in $(seq 1 11);
do
	mkdir -p gpu-accel-CG/densenet201-multithread/${num_thread}thread/
done

## GPU-Accel
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p gpu-accel_jitter/$model/
done

## GPU-Accel_1thread
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p gpu-accel_1thread/$model/
done

## GPU-Accel-MP
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p gpu-accel-mp/$model/
done

## GPU-Accel-MP-Reverse
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p gpu-accel-mp-reverse/$model/
done

## CPU-Reclaiming
for var in {0..305}
	do
	mkdir -p cpu-reclaiming/densenet201/${var}glayer/
	done

for var in {0..37}
	do
	mkdir -p cpu-reclaiming/yolov4-tiny/${var}glayer/
	done

for var in {0..162}
	do
	mkdir -p cpu-reclaiming/yolov4/${var}glayer/
	done

for var in {0..100}
	do
	mkdir -p cpu-reclaiming/yolov7-tiny/${var}glayer/
	done

for var in {0..144}
	do
	mkdir -p cpu-reclaiming/yolov7/${var}glayer/
	done

for var in {0..207}
	do
	mkdir -p cpu-reclaiming/resnet152/${var}glayer/
	done

for var in {0..81}
	do
	mkdir -p cpu-reclaiming/csmobilenet-v2/${var}glayer/
	done

for var in {0..51}
	do
	mkdir -p cpu-reclaiming/squeezenet/${var}glayer/
	done

for var in {0..138}
	do
	mkdir -p cpu-reclaiming/enetb0/${var}glayer/
	done

## CPU-Reclaiming-CRG
for thread in {1..11}; do
	for var in {0..306}
		do
		mkdir -p cpu-reclaiming-CRG/densenet201-multithread/${thread}thread/${var}glayer/
		done
done


for var in {1..305}
	do
	mkdir -p cpu-reclaiming-CRG/densenet201/${var}glayer/
	done

for var in {1..37}
	do
	mkdir -p cpu-reclaiming-CRG/yolov4-tiny/${var}glayer/
	done

for var in {1..162}
	do
	mkdir -p cpu-reclaiming-CRG/yolov4/${var}glayer/
	done

for var in {1..100}
	do
	mkdir -p cpu-reclaiming-CRG/yolov7-tiny/${var}glayer/
	done

for var in {1..144}
	do
	mkdir -p cpu-reclaiming-CRG/yolov7/${var}glayer/
	done

for var in {1..207}
	do
	mkdir -p cpu-reclaiming-CRG/resnet152/${var}glayer/
	done

for var in {1..81}
	do
	mkdir -p cpu-reclaiming-CRG/csmobilenet-v2/${var}glayer/
	done

for var in {1..51}
	do
	mkdir -p cpu-reclaiming-CRG/squeezenet/${var}glayer/
	done

for var in {1..138}
	do
	mkdir -p cpu-reclaiming-CRG/enetb0/${var}glayer/
	done

## CPU-Reclaiming-CRG


for thread in {1..11}; do
	for var in {0..306}
		do
		mkdir -p cpu-reclaiming-GRC/densenet201-multithread/${thread}thread/${var}glayer/
		done
done

for thread in {1..11}; do
	for var in {0..306}
		do
		mkdir -p cpu-reclaiming-GRC/enetb0-multithread/${thread}thread/${var}glayer/
		done
done

for thread in {1..11}; do
	for var in {0..306}
		do
		mkdir -p cpu-reclaiming-GRC/resnet152-multithread/${thread}thread/${var}glayer/
		done
done

for thread in {1..11}; do
	for var in {0..306}
		do
		mkdir -p cpu-reclaiming-GRC/csmobilenet-v2-multithread/${thread}thread/${var}glayer/
		done
done

for var in {0..305}
	do
	mkdir -p cpu-reclaiming-GRC/densenet201/${var}glayer/
	done

for var in {0..37}
	do
	mkdir -p cpu-reclaiming-GRC/yolov4-tiny/${var}glayer/
	done

for var in {0..162}
	do
	mkdir -p cpu-reclaiming-GRC/yolov4/${var}glayer/
	done

for var in {0..100}
	do
	mkdir -p cpu-reclaiming-GRC/yolov7-tiny/${var}glayer/
	done

for var in {0..144}
	do
	mkdir -p cpu-reclaiming-GRC/yolov7/${var}glayer/
	done

for var in {0..207}
	do
	mkdir -p cpu-reclaiming-GRC/resnet152/${var}glayer/
	done

for var in {0..81}
	do
	mkdir -p cpu-reclaiming-GRC/csmobilenet-v2/${var}glayer/
	done

for var in {0..51}
	do
	mkdir -p cpu-reclaiming-GRC/squeezenet/${var}glayer/
	done

for var in {0..138}
	do
	mkdir -p cpu-reclaiming-GRC/enetb0/${var}glayer/
	done

## CPU-Reclaiming-RCG
for thread in {1..11}; do
	for var in {0..306}
		do
		mkdir -p cpu-reclaiming-RCG/densenet201-multithread/${thread}thread/${var}glayer/
		done
done

for var in {1..305}
	do
	mkdir -p cpu-reclaiming-RCG/densenet201/${var}glayer/
	done

for var in {1..37}
	do
	mkdir -p cpu-reclaiming-RCG/yolov4-tiny/${var}glayer/
	done

for var in {1..162}
	do
	mkdir -p cpu-reclaiming-RCG/yolov4/${var}glayer/
	done

for var in {1..100}
	do
	mkdir -p cpu-reclaiming-RCG/yolov7-tiny/${var}glayer/
	done

for var in {1..144}
	do
	mkdir -p cpu-reclaiming-RCG/yolov7/${var}glayer/
	done

for var in {1..207}
	do
	mkdir -p cpu-reclaiming-RCG/resnet152/${var}glayer/
	done

for var in {1..81}
	do
	mkdir -p cpu-reclaiming-RCG/csmobilenet-v2/${var}glayer/
	done

for var in {1..51}
	do
	mkdir -p cpu-reclaiming-RCG/squeezenet/${var}glayer/
	done

for var in {1..138}
	do
	mkdir -p cpu-reclaiming-RCG/enetb0/${var}glayer/
	done

## CPU-Reclaiming-CRG
for var in {1..305}
	do
	mkdir -p cpu-reclaiming-RGC/densenet201/${var}glayer/
	done

for var in {1..37}
	do
	mkdir -p cpu-reclaiming-RGC/yolov4-tiny/${var}glayer/
	done

for var in {1..162}
	do
	mkdir -p cpu-reclaiming-RGC/yolov4/${var}glayer/
	done

for var in {1..100}
	do
	mkdir -p cpu-reclaiming-RGC/yolov7-tiny/${var}glayer/
	done

for var in {1..144}
	do
	mkdir -p cpu-reclaiming-RGC/yolov7/${var}glayer/
	done

for var in {1..207}
	do
	mkdir -p cpu-reclaiming-RGC/resnet152/${var}glayer/
	done

for var in {1..81}
	do
	mkdir -p cpu-reclaiming-RGC/csmobilenet-v2/${var}glayer/
	done

for var in {1..51}
	do
	mkdir -p cpu-reclaiming-RGC/squeezenet/${var}glayer/
	done

for var in {1..138}
	do
	mkdir -p cpu-reclaiming-RGC/enetb0/${var}glayer/
	done

## CPU-Reclaiming-GCR
for thread in {1..11}; do
	for var in {0..306}
		do
		mkdir -p cpu-reclaiming-GCR/densenet201-multithread/${thread}thread/${var}glayer/
		done
done

for var in {1..305}
	do
	mkdir -p cpu-reclaiming-GCR/densenet201/${var}glayer/
	done

for var in {1..37}
	do
	mkdir -p cpu-reclaiming-GCR/yolov4-tiny/${var}glayer/
	done

for var in {1..162}
	do
	mkdir -p cpu-reclaiming-GCR/yolov4/${var}glayer/
	done

for var in {1..100}
	do
	mkdir -p cpu-reclaiming-GCR/yolov7-tiny/${var}glayer/
	done

for var in {1..144}
	do
	mkdir -p cpu-reclaiming-GCR/yolov7/${var}glayer/
	done

for var in {1..207}
	do
	mkdir -p cpu-reclaiming-GCR/resnet152/${var}glayer/
	done

for var in {1..81}
	do
	mkdir -p cpu-reclaiming-GCR/csmobilenet-v2/${var}glayer/
	done

for var in {1..51}
	do
	mkdir -p cpu-reclaiming-GCR/squeezenet/${var}glayer/
	done

for var in {1..138}
	do
	mkdir -p cpu-reclaiming-GCR/enetb0/${var}glayer/
	done

## CPU-Reclaiming-mp
for var in {1..305}
	do
	mkdir -p cpu-reclaiming-mp/densenet201/${var}glayer/
	done

for var in {1..37}
	do
	mkdir -p cpu-reclaiming-mp/yolov4-tiny/${var}glayer/
	done

for var in {1..162}
	do
	mkdir -p cpu-reclaiming-mp/yolov4/${var}glayer/
	done

for var in {1..100}
	do
	mkdir -p cpu-reclaiming-mp/yolov7-tiny/${var}glayer/
	done

for var in {1..144}
	do
	mkdir -p cpu-reclaiming-mp/yolov7/${var}glayer/
	done

for var in {1..207}
	do
	mkdir -p cpu-reclaiming-mp/resnet152/${var}glayer/
	done

for var in {1..81}
	do
	mkdir -p cpu-reclaiming-mp/csmobilenet-v2/${var}glayer/
	done

for var in {1..51}
	do
	mkdir -p cpu-reclaiming-mp/squeezenet/${var}glayer/
	done

for var in {1..138}
	do
	mkdir -p cpu-reclaiming-mp/enetb0/${var}glayer/
	done
	
## GPU-Accel_1thread
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	for var in {0..11}
	do
		mkdir -p gpu-accel_layer_test/${model}-multithread/${var}thread/
	done
done
