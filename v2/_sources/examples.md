# Example oneCCL app

```{eval-rst}
.. note::
    This document describes the new C API that closely follows the NVIDIA Collective Communications Library (NCCL)* API standard. Documentation for the legacy C++ API can be found `here <../index.html>`_.
```

This guide explains:

* How to set up oneCCL with the NCCL-like C API.
* How to build and run a simple GPU-based collective communication example.
* How to integrate MPI and SYCL for process management and GPU execution.

### Prerequisites
Before starting, ensure the following:

* Intel oneAPI Base Toolkit or any other distribution of oneCCL is installed.
* MPI implementation (e.g., Intel MPI or MPICH) is available.
* The C++ compiler supports SYCL (e.g., `icpx` from oneAPI).
* (Optional) A system with multiple GPUs or a multi-node cluster is used.

### 1. Include Required Headers
Add the following headers to your source file:

* `oneapi/ccl.h` for oneCCL C API.
* `mpi.h` for MPI initialization and communication.
* `sycl/sycl.hpp` for GPU operations.
```cpp
#include <iostream>
#include <mpi.h>
#include <oneapi/ccl.h>
#include <sycl/sycl.hpp>
```

### 2. Initialize MPI and oneCCL

* Initialize MPI with multithreading support.
* Retrieve oneCCL version for verification.
* Determine global rank, world size, and local rank.
```cpp
int rank = 0;
int local_rank = 0;
int world_size = 0;
int version = 0;
constexpr int kCount = 16;

MPI_Comm local_comm = 0;
onecclComm_t comm = nullptr;
onecclResult_t result = onecclSuccess;
onecclUniqueId uid;

onecclGetVersion(&version);
std::cout << "Running oneCCL version: " << version << "\n";

MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, nullptr);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &world_size);
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                    &local_comm);
MPI_Comm_rank(local_comm, &local_rank);
```

### 3. Create and Share Unique ID

* Generate a unique ID for the communicator on rank 0 using onecclGetUniqueId.
* Broadcast the ID to all ranks using MPI, so the processes will form one communicator later.
```cpp
if (rank == 0) {
    onecclGetUniqueId(&uid);
}
MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
```

### 4. Associate GPU Device

* Assign each rank to a GPU device using `onecclSetDevice`.
* Create a SYCL queue for GPU execution.
```cpp
auto sycl_queue = create_queue(local_rank);
result = onecclSetDevice(local_rank);
if (result != onecclSuccess) {
    std::cerr << "Failed to set device.\n";
    return 1;
}
```

### 5. Initialize Communicator

* Create a communicator using onecclCommInitRank.
```cpp
result = onecclCommInitRank(&comm, world_size, uid, rank);
if (result != onecclSuccess) {
    std::cerr << "Failed to initialize communicator.\n";
    return 1;
}
```

### 6. Allocate Buffers and Prepare Data

* Allocate GPU memory using SYCL.
* Initialize data with a GPU kernel.
```cpp
int *sendbuff = static_cast<int *>(
    sycl::malloc_device(kCount * sizeof(int), sycl_queue));
int *recvbuff = static_cast<int *>(
    sycl::malloc_device(kCount * sizeof(int), sycl_queue));

sycl_queue.submit([&](sycl::handler &h) {
    h.parallel_for<class prepare_data>(
        sycl::range<1>(kCount), [=](sycl::id<1> idx) {
            sendbuff[idx] += (rank + 1) * 10;
        });
});
```

### 7. Perform a Collective Operation

* Execute `onecclAllReduce` to sum data across ranks.
* Pass a pointer to the SYCL queue as the last argument of the collective operation to ensure proper scheduling of the operation.
```cpp
result = onecclAllReduce(
    sendbuff, recvbuff, kCount, onecclInt, onecclSum, comm, &sycl_queue);
if (result != onecclSuccess) {
    std::cerr << "AllReduce operation failed.\n";
    return 1;
}
```

### 8. Post-Processing

* Compute average on GPU.
* Copy results to host then and print.
```cpp
sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class average>(
        sycl::range<1>(kCount), [=](sycl::id<1> idx) {
            recvbuff[idx] = recvbuff[idx] / world_size;
        });
});

std::vector<int> recvbuff_host(kCount);
sycl_queue.memcpy(recvbuff_host.data(), recvbuff, kCount * sizeof(int))
    .wait();

for (int i = 0; i < recvbuff_host.size(); i++) {
    std::cout << recvbuff_host[i] << " ";
}
std::cout << '\n';
```
### 9. Cleanup

* Destroy communicator and finalize MPI.
```cpp
result = onecclCommDestroy(comm);
if (result != onecclSuccess) {
    std::cerr << "Destroy communicator failed.\n";
    return 1;
}

MPI_Finalize();
```

### Complete code

```cpp
#include <iostream>
#include <mpi.h>
#include <oneapi/ccl.h>
#include <sycl/sycl.hpp>

static sycl::queue create_queue(int local_rank) {
    auto platforms = sycl::platform::get_platforms();
    sycl::platform l0_platform;
    bool l0_found = false;

    for (const auto &platform : platforms) {
        if (platform.get_backend() == sycl::backend::ext_oneapi_level_zero) {
            l0_platform = platform;
            l0_found = true;
            break;
        }
    }

    if (!l0_found) {
        throw std::runtime_error("Level-Zero platform not found.");
    }

    return sycl::queue(
        l0_platform
            .get_devices()[local_rank % l0_platform.get_devices().size()],
        {sycl::property::queue::in_order{},
         sycl::property::queue::enable_profiling{}});
}

int main() {
    int rank = 0;
    int local_rank = 0;
    int world_size = 0;
    int version = 0;
    constexpr int kCount = 16;

    MPI_Comm local_comm = 0;
    onecclComm_t comm = nullptr;
    onecclResult_t result = onecclSuccess;
    onecclUniqueId uid;

    onecclGetVersion(&version);
    std::cout << "Running oneCLL version: " << version << "\n";

    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);

    if (rank == 0) {
        onecclGetUniqueId(&uid);
    }
    MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);

    auto sycl_queue = create_queue(local_rank);
    result = onecclSetDevice(local_rank);
    if (result != onecclSuccess) {
        std::cerr << "Failed to set device.\n";
        return 1;
    }

    result = onecclCommInitRank(&comm, world_size, uid, rank);
    if (result != onecclSuccess) {
        std::cerr << "Failed to initialize communicator.\n";
        return 1;
    }

    int *sendbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), sycl_queue));
    int *recvbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), sycl_queue));

    sycl_queue.submit([&](sycl::handler &h) {
        h.parallel_for<class prepare_data>(
            sycl::range<1>(kCount), [=](sycl::id<1> idx) {
                sendbuff[idx] += (rank + 1) * 10;
            });
    });

    result = onecclAllReduce(
        sendbuff, recvbuff, kCount, onecclInt, onecclSum, comm, &sycl_queue);
    if (result != onecclSuccess) {
        std::cerr << "AllReduce operation failed.\n";
        return 1;
    }

    sycl_queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class average>(
            sycl::range<1>(kCount), [=](sycl::id<1> idx) {
                recvbuff[idx] = recvbuff[idx] / world_size;
            });
    });

    std::vector<int> recvbuff_host(kCount);
    sycl_queue.memcpy(recvbuff_host.data(), recvbuff, kCount * sizeof(int))
        .wait();

    for (int i = 0; i < recvbuff_host.size(); i++) {
        std::cout << recvbuff_host[i] << " ";
    }
    std::cout << '\n';

    result = onecclCommDestroy(comm);
    if (result != onecclSuccess) {
        std::cerr << "Destroy communicator failed.\n";
        return 1;
    }

    MPI_Finalize();
    return 0;
}
```
### Build and run

* To build and execute this app you can use:
```sh
# Setup environment for oneCCL
source <oneCCL install directory>/env/vars.sh

# Compile the example using `icpx` for GPU support
icpx example.cpp -fsycl -lccl -lmpi -o example

# Run the app through MPI
mpiexec -n 2 ./a.out
```
