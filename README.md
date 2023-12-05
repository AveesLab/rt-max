# Data-parallel Framework
Darknet frame rate & delay optimization 

### ※ We use NVIDIA Jetson AGX Orin 64GB, and our code is based on this hardware platform. 

# Setup
## Power Mode Setting
```
# open /etc/nvpmodel.conf to configure power mode
$ sudo gedit /etc/nvpmodel.conf
```

```
# copy and paste below code to nvpmodel configuration file
< POWER_MODEL ID=4 NAME=GPU_FREQ_100 >
CPU_ONLINE CORE_0 1
CPU_ONLINE CORE_1 1
CPU_ONLINE CORE_2 1
CPU_ONLINE CORE_3 1
CPU_ONLINE CORE_4 1
CPU_ONLINE CORE_5 1
CPU_ONLINE CORE_6 1
CPU_ONLINE CORE_7 1
CPU_ONLINE CORE_8 1
CPU_ONLINE CORE_9 1
CPU_ONLINE CORE_10 1
CPU_ONLINE CORE_11 1
TPC_POWER_GATING TPC_PG_MASK 0
GPU_POWER_CONTROL_ENABLE GPU_PWR_CNTL_EN on
CPU_A78_0 MIN_FREQ -1
CPU_A78_0 MAX_FREQ -1
CPU_A78_1 MIN_FREQ -1
CPU_A78_1 MAX_FREQ -1
CPU_A78_2 MIN_FREQ -1
CPU_A78_2 MAX_FREQ -1
GPU MIN_FREQ 1224000000
GPU MAX_FREQ 1300500000
GPU_POWER_CONTROL_DISABLE GPU_PWR_CNTL_DIS auto
EMC MAX_FREQ 0
DLA0_CORE MAX_FREQ -1
DLA1_CORE MAX_FREQ -1
DLA0_FALCON MAX_FREQ -1
DLA1_FALCON MAX_FREQ -1
PVA0_VPS MAX_FREQ -1
PVA0_AXI MAX_FREQ -1
```

## OpenBLAS
```
$ git clone https://github.com/AveesLab/OpenBLAS.git
$ cd OpenBLAS
$ ./setup.sh
```

# Implementation

## Sequential Architecture
```
# use GPU as Inference
./sequential_test.sh -model {model} -isGPU 1

# use CPU as Inference
./sequential_test.sh -model {model} -isGPU 0
```

## Pipeline Architecture
```
# use GPU as Inference
./pipeline_test.sh -model {model} -isGPU 1

# use CPU as Inference
./pipeline_test.sh -model {model} -isGPU 0
```

## Data-Parallel Architecture
```
./data_parallel_test.sh -model {model}
```

## Partial DNN Acceleration Architecture
```
./gpu_accel_test.sh -model {model}
```
