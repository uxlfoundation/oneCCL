# Example oneCCL app

```{eval-rst}
.. note::
    In oneCCL version 2021.17 included with the 2025.3 oneAPI release, oneCCL has added support for a new **C API** that closely follows the NVIDIA Collective Communications Libary (NCCL)* API standard. Details about the new API, instructions on how to build, and run an example can be found `here <./index.html>`_.

    The existing C++ API will remain the default API for the 2021.17 release and can be found `here <../index.html>`_.
```

This example showcases a simple app using oneCCL for communication across GPUs. The example uses MPI as process launcher, but a different process launcher can also be used with oneCCL.

The first step to use **NCCL-like C API** for oneCCL is to include `oneapi/ccl.h` header. `mpi.h` is included as the example uses **MPI** to broadcast the uniqueId needed to build a oneCCL communicator. `sycl/sycl.hpp` is included to execute GPU kernels.
```cpp
#include <iostream>
#include <mpi.h>
#include <oneapi/ccl.h>
#include <sycl/sycl.hpp>
```

Next we will declre a few variables that will be used later in the example.
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
```

The first step required to setup `onecclComm_t` is the creation of `onecclUniqueId`. One rank calls `onecclGetUniqueId` and then broadcast the `uniqueId` to all the ranks that will participate in the communicator. This example `MPI_Bcast`, but applications can use a different method.
```cpp
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
```

Before creating a communicator, each rank needs to be associated with a GPU device. This can be done by calling `onecclSetDevice` before `onecclCommInitRank`. This example uses a simple helper function `create_queue` to create a `sycl::queue` based on `local_rank`, so process with local index 0 will be using GPU at index 0, and so on.
```cpp
auto sycl_queue = create_queue(local_rank);
result = onecclSetDevice(local_rank);
if (result != onecclSuccess) {
    std::cerr << "Failed to set device.\n";
    return 1;
}
```

Now the communicator can be created.
```cpp
result = onecclCommInitRank(&comm, world_size, uid, rank);
if (result != onecclSuccess) {
    std::cerr << "Failed to initialize communicator.\n";
    return 1;
}
```

Next, the example allocates GPU buffers using `SYCL` APIs and submits a simple kernel to setup their content.
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

Now `onecclAllReduce` can execute on the created communicator. Note that the last argument of `onecclAllReduce` is a pointer to `sycl::queue`. This is required to execute collectives on the GPU and properly schedule them with respect to other kernels submitted to the same GPU and SYCL queue.
```cpp
result = onecclAllReduce(
    sendbuff, recvbuff, kCount, onecclInt, onecclSum, comm, &sycl_queue);
if (result != onecclSuccess) {
    std::cerr << "AllReduce operation failed.\n";
    return 1;
}
```

As a final step, this example application submits a kernel to compute the average and copy back to the host.
Finally, the call to `onecclCommDestroy` frees the resources allocated to the communicator
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

result = onecclCommDestroy(comm);
if (result != onecclSuccess) {
    std::cerr << "Destroy communicator failed.\n";
    return 1;
}

MPI_Finalize();
```

Here's complete code for the example, feel free to experiment with that!
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

To build and execute this app you can use:
```sh
# Setup environment for oneCCL
source <oneCCL install directory>/env/vars.sh

# Compile the example using `icpx` for GPU support
icpx example.cpp -fsycl -lccl -lmpi -o example

# Run the app through MPI
mpiexec -n 2 ./a.out
```
