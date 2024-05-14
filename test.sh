# ORIN 1
./gpu_accel_GC_test.sh -model densenet201 -num_thread 14
./gpu_accel_GC_test.sh -model densenet201 -num_thread 13
./gpu_accel_GC_test.sh -model densenet201 -num_thread 12
./gpu_accel_GC_test.sh -model densenet201 -num_thread 11
./gpu_accel_GC_test.sh -model densenet201 -num_thread 10

./cpu_reclaiming_GRC_test.sh -model densenet201 -num_thread 14
./cpu_reclaiming_GRC_test.sh -model densenet201 -num_thread 13
./cpu_reclaiming_GRC_test.sh -model densenet201 -num_thread 12
./cpu_reclaiming_GRC_test.sh -model densenet201 -num_thread 11
./cpu_reclaiming_GRC_test.sh -model densenet201 -num_thread 10
