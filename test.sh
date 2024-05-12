./gpu_accel_CG_test.sh -model densenet201 -num_thread 11
./gpu_accel_CG_test.sh -model densenet201 -num_thread 10
./gpu_accel_CG_test.sh -model densenet201 -num_thread 9

./gpu_accel_CG_test.sh -model densenet201 -num_thread 8
./gpu_accel_CG_test.sh -model densenet201 -num_thread 7
./gpu_accel_CG_test.sh -model densenet201 -num_thread 6

./gpu_accel_CG_test.sh -model densenet201 -num_thread 5
./gpu_accel_CG_test.sh -model densenet201 -num_thread 4
./gpu_accel_CG_test.sh -model densenet201 -num_thread 3
./gpu_accel_CG_test.sh -model densenet201 -num_thread 2
./gpu_accel_CG_test.sh -model densenet201 -num_thread 1


./cpu_reclaiming_CRG_test.sh -model densenet201 -num_thread 10
./cpu_reclaiming_CRG_test.sh -model densenet201 -num_thread 9
./cpu_reclaiming_CRG_test.sh -model densenet201 -num_thread 8


./cpu_reclaiming_GCR_test.sh -model densenet201 -num_thread 10
./cpu_reclaiming_GCR_test.sh -model densenet201 -num_thread 9
./cpu_reclaiming_GCR_test.sh -model densenet201 -num_thread 8
