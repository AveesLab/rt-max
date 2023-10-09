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

touch sequential_cpu_utilization.csv
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	cp sequential_cpu_utilization.csv sequential/$model/
done
rm sequential_cpu_utilization.csv

## Sequential-multiblas
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p sequential-multiblas/$model/
done

touch sequential_cpu_utilization.csv
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	cp sequential_cpu_utilization.csv sequential-multiblas/$model/
done
rm sequential_cpu_utilization.csv

## Pipeline
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p pipeline/$model/
done

touch pipeline_cpu_utilization.csv
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	cp pipeline_cpu_utilization.csv pipeline/$model/
done
rm pipeline_cpu_utilization.csv

## Data-Parallel
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p data-parallel/$model/
done

touch data-parallel_cpu_utilization.csv
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	cp data-parallel_cpu_utilization.csv data-parallel/$model/
done
rm data-parallel_cpu_utilization.csv

## Data-Parallel-MP
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p data-parallel-mp/$model/
done

touch data-parallel-mp_cpu_utilization.csv
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	cp data-parallel-mp_cpu_utilization.csv data-parallel-mp/$model/
done
rm data-parallel-mp_cpu_utilization.csv

## GPU-Accel
for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p gpu-accel/$model/
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

## CPU-Reclaiming
for var in {1..305}
	do
	mkdir -p cpu-reclaiming/densenet201/${var}glayer/
	done

for var in {1..37}
	do
	mkdir -p cpu-reclaiming/yolov4-tiny/${var}glayer/
	done

for var in {1..162}
	do
	mkdir -p cpu-reclaiming/yolov4/${var}glayer/
	done

for var in {1..100}
	do
	mkdir -p cpu-reclaiming/yolov7-tiny/${var}glayer/
	done

for var in {1..144}
	do
	mkdir -p cpu-reclaiming/yolov7/${var}glayer/
	done

for var in {1..207}
	do
	mkdir -p cpu-reclaiming/resnet152/${var}glayer/
	done

for var in {1..81}
	do
	mkdir -p cpu-reclaiming/csmobilenet/${var}glayer/
	done

for var in {1..51}
	do
	mkdir -p cpu-reclaiming/squeezenet/${var}glayer/
	done

for var in {1..138}
	do
	mkdir -p cpu-reclaiming/enetb0/${var}glayer/
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
	mkdir -p cpu-reclaiming-mp/csmobilenet/${var}glayer/
	done

for var in {1..51}
	do
	mkdir -p cpu-reclaiming-mp/squeezenet/${var}glayer/
	done

for var in {1..138}
	do
	mkdir -p cpu-reclaiming-mp/enetb0/${var}glayer/
	done
