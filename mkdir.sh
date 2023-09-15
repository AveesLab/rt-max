mkdir measure
cd measure
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

for model in "yolov4" "yolov4-tiny" "yolov7" "yolov7-tiny" "densenet201" "resnet152" "csmobilenet-v2" "squeezenet" "enetb0"
do
	mkdir -p gpu-accel/$model/
done

for var in {1..305}
	do
	mkdir -p reclaiming/densenet201/${var}glayer/
	done

for var in {1..100}
	do
	mkdir -p reclaiming/yolov7-tiny/${var}glayer/
	done

for var in {1..144}
	do
	mkdir -p reclaiming/yolov7/${var}glayer/
	done

for var in {1..207}
	do
	mkdir -p reclaiming/resnet152/${var}glayer/
	done

for var in {1..81}
	do
	mkdir -p reclaiming/csmobilenet/${var}glayer/
	done

for var in {1..51}
	do
	mkdir -p reclaiming/squeezenet/${var}glayer/
	done

for var in {1..138}
	do
	mkdir -p reclaiming/enetb0/${var}glayer/
	done
