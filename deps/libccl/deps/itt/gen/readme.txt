Build and install:
./update_itt_calls.sh

Usage with vtune:
LD_PRELOAD=${CCL_REPO}/deps/itt/lib64/tracing_functions.so vtune ...

Example:
CCL_ITT_LEVEL=1 LD_PRELOAD=${CCL_REPO}/deps/itt/lib64/tracing_functions.so  mpiexec -n 2 vtune -collect gpu-offload -knob enable-stack-collection=true --run-pass-thru=--sample-after-multiplier=0.1 -knob collect-programming-api=true -r ./bench -- ./examples/benchmark/benchmark -l allreduce -i 10 -y 8388608 -p 1 -n 1 -b sycl --iter_policy off -c off
