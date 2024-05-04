sudo sysctl -w kernel.sched_rt_runtime_us=-1
sudo ldconfig /usr/local/lib
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
sudo -E chrt -f 99 ./test.sh
