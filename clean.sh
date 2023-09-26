cd measure

## Sequential
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	find sequential/$model/ -type f -exec rm -f {} +
done

## Pipeline
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	find pipeline/$model/ -type f -exec rm -f {} +
	
done


## Data-Parallel
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	find data-parallel/$model/ -type f -exec rm -f {} +
	
done

## Data-Parallel-MP
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	find data-parallel-mp/$model/ -type f -exec rm -f {} +
	
done

## GPU-Accel
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	find gpu-accel/$model/ -type f -exec rm -f {} +
	
done

## GPU-Accel-MP
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	find gpu-accel-mp/$model/ -type f -exec rm -f {} +
	
done

## CPU-Reclaiming
for var in {1..305}
	do
	find cpu-reclaiming/densenet201/${var}glayer/ -type f -exec rm -f {} +
	
	done

for var in {1..37}
	do
	find cpu-reclaiming/yolov4-tiny/${var}glayer/ -type f -exec rm -f {} +
	
	done

for var in {1..162}
	do
	find cpu-reclaiming/yolov4/${var}glayer/ -type f -exec rm -f {} +
	
	done

for var in {1..100}
	do
	find cpu-reclaiming/yolov7-tiny/${var}glayer/ -type f -exec rm -f {} +
	
	done

for var in {1..144}
	do
	find cpu-reclaiming/yolov7/${var}glayer/ -type f -exec rm -f {} +
	
	done

for var in {1..207}
	do
	find cpu-reclaiming/resnet152/${var}glayer/ -type f -exec rm -f {} +
	
	done

for var in {1..81}
	do
	find cpu-reclaiming/csmobilenet/${var}glayer/ -type f -exec rm -f {} +
	
	done

for var in {1..51}
	do
	find cpu-reclaiming/squeezenet/${var}glayer/ -type f -exec rm -f {} +
	
	done

for var in {1..138}
	do
	find cpu-reclaiming/enetb0/${var}glayer/ -type f -exec rm -f {} +
	
	done

## CPU-Reclaiming-mp
for var in {1..305}
	do
	find cpu-reclaiming-mp/densenet201/${var}glayer/ -type f -exec rm -f {} +
	
	done

for var in {1..37}
	do
	find cpu-reclaiming-mp/yolov4-tiny/${var}glayer/ -type f -exec rm -f {} +
	
	done

for var in {1..162}
	do
	find cpu-reclaiming-mp/yolov4/${var}glayer/ -type f -exec rm -f {} +
	
	done

for var in {1..100}
	do
	find cpu-reclaiming-mp/yolov7-tiny/${var}glayer/ -type f -exec rm -f {} +
	
	done

for var in {1..144}
	do
	find cpu-reclaiming-mp/yolov7/${var}glayer/ -type f -exec rm -f {} +
	
	done

for var in {1..207}
	do
	find cpu-reclaiming-mp/resnet152/${var}glayer/ -type f -exec rm -f {} +
	
	done

for var in {1..81}
	do
	find cpu-reclaiming-mp/csmobilenet/${var}glayer/ -type f -exec rm -f {} +
	
	done

for var in {1..51}
	do
	find cpu-reclaiming-mp/squeezenet/${var}glayer/ -type f -exec rm -f {} +
	
	done

for var in {1..138}
	do
	find cpu-reclaiming-mp/enetb0/${var}glayer/ -type f -exec rm -f {} +
	
	done
