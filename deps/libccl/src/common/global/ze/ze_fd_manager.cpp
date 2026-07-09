/*
 Copyright 2016-2026 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include "common/global/global.hpp"
#include "common/global/ze/ze_fd_manager.hpp"
#include "common/log/log.hpp"
#include "common/utils/exchange_utils.hpp"
#include "common/utils/utils.hpp"
#include "common/utils/yield.hpp"

#ifdef CCL_ENABLE_DRM
#include "i915_drm.h"
#endif // CCL_ENABLE_DRM

// pidfd system calls
#ifndef __NR_pidfd_open
#define __NR_pidfd_open 434
#endif // __NR_pidfd_open
#ifndef __NR_pidfd_getfd
#define __NR_pidfd_getfd 438
#endif // __NR_pidfd_getfd

#include <errno.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/socket.h>
#include <sys/un.h>

namespace ccl {
namespace ze {

fd_manager::fd_manager(std::shared_ptr<atl_base_comm> comm) : comm(comm) {
    CCL_THROW_IF_NOT(comm, "no comm in fd_manager init");
    device_fds = init_device_fds();
    exchange_device_fds();
    LOG_DEBUG("init completed");
}

fd_manager::~fd_manager() {
    all_socks.clear();
    all_pids.clear();
    for (auto fd : device_fds) {
        ccl::utils::close_fd(fd);
    }
    device_fds.clear();
    device_bdfs.clear();
}

bool fd_manager::is_pidfd_supported() {
    int pid = getpid();
    char filename[] = "/tmp/oneccl_pidfd_check_getXXXXXXXXXX";
    std::vector<int> fds;
    bool result = true;

    auto check_fd = [&](int fd) {
        if (fd == ccl::utils::invalid_fd) {
            result = false;
        }
        fds.push_back(fd);
    };

    mode_t prev_umask = umask(rwe_umask);

    int file_fd = mkstemp(filename);
    check_fd(file_fd);

    umask(prev_umask);

    int pidfd = syscall(__NR_pidfd_open, pid, 0);
    check_fd(pidfd);

    int dupfd = syscall(__NR_pidfd_getfd, pidfd, file_fd, 0);
    check_fd(dupfd);

    for (auto &fd : fds) {
        ccl::utils::close_fd(fd);
    }
    unlink(filename);
    return result;
}

// get functions impl
std::vector<int> fd_manager::get_device_fds() {
    return device_fds;
}

std::vector<device_bdf_info> fd_manager::get_physical_devices() {
    return physical_devices;
}

#ifdef ZE_PCI_PROPERTIES_EXT_NAME
int fd_manager::get_physical_device_idx(std::vector<device_bdf_info> devs,
                                        ze_pci_address_ext_t pci) {
    // the purpose is to get the device idx based on BDF,
    // First, we need to discover and parse the BDF values from the string
    // representation of the BDF.
    // one way to do this is to use the strtoul function to convert
    // the BDF values from the string representation to integers.
    // init_device_bdfs(), find_device_idx(), find_bdf()

    // The BDF values can then be used to identify the specific
    // device by accessing the relevant fields in a device-specific
    // data structure (device_bdf_info), such as the PCI
    // configuration space for PCI devices, here we fill the final structure
    int physical_idx = ccl::ze::fd_manager::invalid_device_idx;
    CCL_THROW_IF_NOT(devs.size() != 0, "unexpected size to init bdfs: ", devs.size());
    for (size_t idx = 0; idx < devs.size(); idx++) {
        if (devs[idx].bdf.domain == pci.domain && devs[idx].bdf.bus == pci.bus &&
            devs[idx].bdf.device == pci.device && devs[idx].bdf.function == pci.function) {
            LOG_DEBUG("get physical device idx: ",
                      idx,
                      ", fd: ",
                      devs[idx].fd,
                      ", domain: ",
                      devs[idx].bdf.domain,
                      ", bus: ",
                      devs[idx].bdf.bus,
                      ", device: ",
                      devs[idx].bdf.device,
                      ", function: ",
                      devs[idx].bdf.function);
            physical_idx = idx;
            break;
        }
    }
    CCL_THROW_IF_NOT(physical_idx != ccl::ze::fd_manager::invalid_physical_idx,
                     "invalid physical_idx: ",
                     physical_idx);
    return physical_idx;
}
#endif // ZE_PCI_PROPERTIES_EXT_NAME

// init functions impl
std::vector<int> fd_manager::init_device_fds() {
    const auto device_dir = ccl::global_data::env().drmfd_dev_render_dir_path.c_str();
    const auto suffix = ccl::global_data::env().drmfd_dev_render_suffix.c_str();

    char device_name[NAME_MAX];
    struct dirent *ent = nullptr;
    std::vector<std::string> device_names;

    DIR *dir = opendir(device_dir);
    CCL_THROW_IF_NOT(dir, "opendir failed: could not open device directory");
    dir_raii dir_obj(dir);

    LOG_DEBUG("search for all devices in the device directory");
    while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_name[0] == '.' || strstr(ent->d_name, suffix) == nullptr) {
            continue;
        }
        memset(device_name, 0, sizeof(device_name));
        int ret = snprintf(device_name, NAME_MAX, "%s%s", device_dir, ent->d_name);
        CCL_THROW_IF_NOT(ret > 0 || ret <= NAME_MAX, "could not create device name");
        device_names.push_back(device_name);
    }

    return fill_device_fds(device_names);
}

std::vector<bdf_info> fd_manager::init_device_bdfs(const size_t size) {
    std::vector<bdf_info> bdfs(0);
#ifdef ZE_PCI_PROPERTIES_EXT_NAME
    const auto device_dir = ccl::global_data::env().drmfd_dev_render_dir_path.c_str();
    const auto suffix = ccl::global_data::env().drmfd_dev_render_suffix.c_str();

    char device_name[NAME_MAX];
    struct dirent *ent = nullptr;
    int idx = 0;
    bdfs.resize(size);

    DIR *dir = opendir(device_dir);
    CCL_THROW_IF_NOT(dir, "bdfs opendir failed: could not open device directory");
    dir_raii dir_obj(dir);

    LOG_DEBUG("BDF search for all devices in the device directory");
    while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_name[0] == '.' || strstr(ent->d_name, suffix) == nullptr) {
            continue;
        }
        memset(device_name, 0, sizeof(device_name));

        int ret = snprintf(device_name, NAME_MAX, "%s%s", device_dir, ent->d_name);
        CCL_THROW_IF_NOT(ret > 0 || ret <= NAME_MAX, "could not create device name");

        bdf_info info{};
        find_bdf(device_name, info);
        bdfs[idx].domain = info.domain;
        bdfs[idx].bus = info.bus;
        bdfs[idx].device = info.device;
        bdfs[idx].function = info.function;
        idx++;
    }

    qsort(&bdfs[0], bdfs.size(), sizeof(bdf_info), fd_manager::compare_bdf);
    LOG_DEBUG("sorted bdf size: ", bdfs.size());
#endif // ZE_PCI_PROPERTIES_EXT_NAME
    return bdfs;
}

// fill functions impl
std::vector<int> fd_manager::fill_device_fds(const std::vector<std::string> &dev_names) {
    CCL_THROW_IF_NOT(!dev_names.empty(), "device names are empty");
    std::vector<int> fds(dev_names.size());
    device_bdfs = init_device_bdfs(dev_names.size());

    int dev_idx = ccl::ze::fd_manager::invalid_fd_idx;
    for (size_t dev_name_idx = 0; dev_name_idx < dev_names.size(); dev_name_idx++) {
        dev_idx = find_device_idx(dev_names[dev_name_idx], dev_name_idx, dev_names.size());

        CCL_THROW_IF_NOT(dev_idx != invalid_device_idx, "invalid_device_idx: ", dev_idx);
        fds[dev_idx] = open(dev_names[dev_name_idx].data(), O_RDWR);
        CCL_THROW_IF_NOT(
            fds[dev_idx] > 0, "open failed: fd: ", fds[dev_idx], ", errno: ", strerror(errno));
        CCL_THROW_IF_NOT(
            fds[dev_idx] != ccl::utils::invalid_fd, "unexpected device fd: ", fds[dev_idx]);
        LOG_DEBUG("device_name: ", dev_names[dev_name_idx], " device_fd: ", fds[dev_idx]);
    }

    CCL_THROW_IF_NOT(!fds.empty(), "fds is empty");
    LOG_DEBUG("completed, fds size: ", fds.size());
    return fds;
}

std::vector<device_bdf_info> fd_manager::fill_physical_devices() {
    std::vector<device_bdf_info> pdevs(device_fds.size());
    CCL_THROW_IF_NOT(!pdevs.empty(), "can not fill physical device, it is empty");
    for (size_t i = 0; i < pdevs.size(); i++) {
        pdevs[i].fd = device_fds[i];
#ifdef ZE_PCI_PROPERTIES_EXT_NAME
        pdevs[i].bdf.domain = device_bdfs[i].domain;
        pdevs[i].bdf.bus = device_bdfs[i].bus;
        pdevs[i].bdf.device = device_bdfs[i].device;
        pdevs[i].bdf.function = device_bdfs[i].function;
#endif // ZE_PCI_PROPERTIES_EXT_NAME
        LOG_DEBUG("physical devices fd: ",
                  pdevs[i].fd,
                  "domain: ",
                  pdevs[i].bdf.domain,
                  "bus: ",
                  pdevs[i].bdf.bus,
                  "device: ",
                  pdevs[i].bdf.device,
                  "function: ",
                  pdevs[i].bdf.function);
    }
    return pdevs;
}

// find functions impl
int fd_manager::compare_bdf(const void *_a, const void *_b) {
    int *a = (int *)_a;
    int *b = (int *)_b;
    int result = memcmp(a, b, sizeof(bdf_info));
    return result ? result : 0;
}

void fd_manager::find_bdf(std::string dev_name, bdf_info &info) {
#ifdef ZE_PCI_PROPERTIES_EXT_NAME
    char *endptr = NULL;
    const char *device_suffix = "-render";

    // it first looks for a device_suffix string in the dev_name.data()
    char *dev_suff_pos = strstr(dev_name.data(), device_suffix);
    CCL_THROW_IF_NOT(dev_suff_pos, "could not find the device suffix position");

    // 12 characters from this position to get a string
    // that represents the BDF values in hexadecimal format
    char *bdf_str_start_ptr = dev_suff_pos - bdf_start_pos;

    // then it uses strtoul function to convert these values
    // from hexadecimal to decimal and store them in the info struct
    // endptr is used to store the address of the next character
    // after the first valid integer converted by strtoul,
    // it'll be updated each time strtoul is called.
    info.domain = strtoul(bdf_str_start_ptr, &endptr, fd_manager::hexadecimal_base);
    info.bus = strtoul(endptr + 1, &endptr, fd_manager::hexadecimal_base);
    info.device = strtoul(endptr + 1, &endptr, fd_manager::hexadecimal_base);
    info.function = strtoul(endptr + 1, &endptr, fd_manager::hexadecimal_base);
    LOG_DEBUG("finding bdf finished");
#endif // ZE_PCI_PROPERTIES_EXT_NAME
}

int fd_manager::find_device_idx(std::string dev_name,
                                const size_t dev_name_idx,
                                const size_t size) {
    auto dev_idx = ccl::ze::fd_manager::invalid_fd_idx;
#ifdef ZE_PCI_PROPERTIES_EXT_NAME
    // find device idx based on bdf
    bdf_info info{};
    find_bdf(dev_name, info);
    for (size_t i = 0; i < size; i++) {
        if (info.domain == device_bdfs[i].domain && info.bus == device_bdfs[i].bus &&
            info.device == device_bdfs[i].device && info.function == device_bdfs[i].function) {
            dev_idx = i;
            LOG_DEBUG("fill dev_idx: ",
                      dev_idx,
                      ", domain: ",
                      device_bdfs[i].domain,
                      ", bus: ",
                      device_bdfs[i].bus,
                      ", device: ",
                      device_bdfs[i].device,
                      ", function: ",
                      device_bdfs[i].function);
            break;
        }
    }
#else // ZE_PCI_PROPERTIES_EXT_NAME
    dev_idx = dev_name_idx;
#endif // ZE_PCI_PROPERTIES_EXT_NAME
    return dev_idx;
}

int fd_manager::pidfd_open(const int pid) {
    int fd = syscall(__NR_pidfd_open, pid, 0);
    CCL_THROW_IF_NOT(fd != ccl::utils::invalid_pid,
                     "pidfd_open failed: fd: ",
                     fd,
                     ", pid: ",
                     pid,
                     ", errno: ",
                     strerror(errno));
    LOG_DEBUG("pidfd_open: pid: ", pid, ", fd: ", fd);
    return fd;
}

int fd_manager::fd_to_mem_handle(int dev_fd, int fd) {
#ifdef CCL_ENABLE_DRM
    struct drm_prime_handle req = { 0, 0, 0 };
    req.fd = fd;

    int ret = ioctl(dev_fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &req);
    CCL_THROW_IF_NOT(ret != ccl::utils::invalid_err_code,
                     "ioctl failed: ret: ",
                     ret,
                     ", errno: ",
                     strerror(errno),
                     ", dev_fd: ",
                     dev_fd,
                     ", fd: ",
                     fd);
    LOG_DEBUG("dev_fd: ", dev_fd, ", req.fd: ", req.fd, ", handle: ", req.handle);
    return req.handle;
#else // CCL_ENABLE_DRM
    return ccl::utils::invalid_mem_handle;
#endif // CCL_ENABLE_DRM
}

int fd_manager::convert_fd_drmfd(int convert_from_fd, int handle) {
#ifdef CCL_ENABLE_DRM
    struct drm_prime_handle req = { 0, 0, 0 };
    req.flags = DRM_CLOEXEC | DRM_RDWR;
    req.handle = handle;

    int ret = ioctl(convert_from_fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &req);
    CCL_THROW_IF_NOT(ret != ccl::utils::invalid_err_code,
                     "ioctl failed: ret: ",
                     ret,
                     ", errno: ",
                     strerror(errno),
                     ", dev_fd: ",
                     convert_from_fd,
                     ", handle: ",
                     handle);
    LOG_DEBUG("drm: dev_fd: ", convert_from_fd, ", req.handle: ", handle, ", fd: ", req.fd);
    return req.fd;
#else // CCL_ENABLE_DRM
    return ccl::utils::invalid_fd;
#endif // CCL_ENABLE_DRM
}

int fd_manager::convert_fd_pidfd(int convert_from_fd, int handle) {
    int fd = syscall(__NR_pidfd_getfd, convert_from_fd, handle, 0);
    if (errno == EPERM) {
        LOG_WARN(strerror(errno),
                 ": This error occurs when the calling process does not have the necessary"
                 " permissions to perform the operation on the target process's file descriptor."
                 " i.e. Check permissions and /proc/sys/kernel/yama/ptrace_scope.")
    }
    CCL_THROW_IF_NOT(fd != ccl::utils::invalid_fd,
                     "pidfd_getfd failed: "
                     "convert_from_fd: ",
                     convert_from_fd,
                     ", fd: ",
                     fd,
                     ", handle: ",
                     handle,
                     ", errno: ",
                     strerror(errno));
    LOG_DEBUG(
        "pidfd_getfd: convert_from_fd: ", convert_from_fd, ", handle: ", handle, ", fd: ", fd);
    return fd;
}

int fd_manager::mem_handle_to_fd(int convert_from_fd, int handle) {
    int fd = ccl::utils::invalid_fd;
    if (ccl::global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::drmfd) {
        fd = convert_fd_drmfd(convert_from_fd, handle);
    }
    else if (ccl::global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::pidfd) {
        fd = convert_fd_pidfd(convert_from_fd, handle);
    }
    else if (ccl::global_data::env().ze_ipc_exchange != ccl::ze::ipc_exchange_mode::none) {
        CCL_THROW("unexpected ipc_exchange_mode");
    }
    return fd;
}

std::vector<int> fd_manager::setup_device_fds(int comm_size,
                                              int rank_idx,
                                              std::vector<bdf_info> &return_bdf) {
    std::vector<int> fds;
    std::vector<bdf_info> bdf_data;
    // bdf_info info;
    if (rank_idx == 0) {
        fds = device_fds;
        return_bdf = device_bdfs;
        // send the fds to all other local processes
        for (int p_idx = 1; p_idx < comm_size; p_idx++) {
            for (auto &fd : fds) {
                ccl::utils::sendmsg_call(
                    all_socks[p_idx],
                    fd,
                    device_bdfs.empty() ? nullptr : device_bdfs.data(),
                    device_bdfs.empty() ? 0 : device_bdfs.size() * sizeof(bdf_info),
                    rank_idx);
            }
        }
    }
    else {
        // receive the fds from local process 0
        for (auto fd : device_fds) {
            ccl::utils::close_fd(fd);
        }
        fds.resize(device_fds.size());
        for (auto &fd : fds) {
            bdf_data.resize(device_bdfs.size());
            ccl::utils::recvmsg_call(all_socks[0],
                                     &fd,
                                     bdf_data.empty() ? nullptr : bdf_data.data(),
                                     bdf_data.empty() ? 0 : bdf_data.size() * sizeof(bdf_info),
                                     rank_idx);
            return_bdf = bdf_data;
        }
    }
    return fds;
}

void fd_manager::exchange_device_fds() {
    int sock_err;
    std::string sock_name;
    struct sockaddr_un sockaddr;
    memset(&sockaddr, 0, sizeof(sockaddr));
    unsigned int sockaddr_len = sizeof(sockaddr);
    int enable = 1;
    int ack = 1;

    int comm_size = comm->get_size();
    int rank_idx = comm->get_rank();

    std::vector<int> all_ints(comm_size, ccl::utils::invalid_fd);
    all_socks.resize(comm_size, ccl::utils::invalid_fd);
    all_pids.resize(comm_size, ccl::utils::invalid_pid);

    pid_t pid = getpid();
    ccl::utils::allgather(comm, &pid, all_pids.data(), sizeof(pid_t));

    CCL_THROW_IF_NOT(!all_pids.empty(), "all_pids shouldn't be empty");
    LOG_DEBUG("pid exchange is done: ", all_pids.size());

    // create a named socket between rank_idx
    // 0 and all other local processes
    if (rank_idx == 0) {
        // TODO: barrier from ofi transport is hanging, remove allgather
        // after fix, here and everywhere
        ccl::utils::allgather(comm, &ack, all_ints.data(), sizeof(int));
        for (int i = 1; i < comm_size; ++i) {
            std::string remote_sock_name;
            struct sockaddr_un remote_sockaddr;

            remote_sock_name = "/tmp/ccl-ipc-fd-sock-" + std::to_string(all_pids[i]) + ":" +
                               std::to_string(i) + "-" + std::to_string(rank_idx) + "-" +
                               std::to_string(comm_size);
            sock_name = "/tmp/ccl-ipc-fd-sock-" + std::to_string(pid) + ":" +
                        std::to_string(rank_idx) + "-" + std::to_string(i) + "-" +
                        std::to_string(comm_size);

            // create a socket for local proc j
            all_socks[i] = socket(AF_UNIX, SOCK_STREAM, 0);
            CCL_THROW_IF_NOT(all_socks[i] != ccl::utils::invalid_fd,
                             "socket failed: sock_err: ",
                             all_socks[i],
                             ", errno: ",
                             strerror(errno));

            if (setsockopt(all_socks[i], SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) ==
                ccl::utils::invalid_err_code) {
                CCL_THROW("setsockopt failed: sock: ", all_socks[i], ", errno: ", strerror(errno));
            }

            sockaddr.sun_family = AF_UNIX;
            strncpy(sockaddr.sun_path, sock_name.c_str(), sizeof(sockaddr.sun_path) - 1);

            sock_err = bind(all_socks[i], (struct sockaddr *)&sockaddr, sockaddr_len);
            CCL_THROW_IF_NOT(sock_err != ccl::utils::invalid_err_code,
                             "bind failed: sock_err: ",
                             sock_err,
                             ", errno: ",
                             strerror(errno));

            // connect to remote socket for local proc j
            remote_sockaddr.sun_family = AF_UNIX;
            strncpy(remote_sockaddr.sun_path,
                    remote_sock_name.c_str(),
                    sizeof(remote_sockaddr.sun_path) - 1);

            sock_err = connect(all_socks[i], (struct sockaddr *)&remote_sockaddr, sockaddr_len);
            if (sock_err < 0) {
                if (errno == ECONNREFUSED || errno == ENOENT) {
                    return;
                }
                CCL_THROW("connect failed: error: ",
                          sock_err,
                          ", errno: ",
                          strerror(errno),
                          ", sock_name: ",
                          sock_name);
            }
        }
    }
    else {
        int sock;
        // create the local socket name
        sock_name = "/tmp/ccl-ipc-fd-sock-" + std::to_string(pid) + ":" + std::to_string(rank_idx) +
                    "-" + std::to_string(0) + "-" + std::to_string(comm_size);
        // create a socket for local proc i
        sock = socket(AF_UNIX, SOCK_STREAM, 0);
        CCL_THROW_IF_NOT(sock != ccl::utils::invalid_fd,
                         "socket failed: sock: ",
                         sock,
                         ", errno: ",
                         strerror(errno));

        if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) ==
            ccl::utils::invalid_err_code) {
            CCL_THROW("setsockopt failed: sock: ", sock, ", errno: ", strerror(errno));
        }
        sockaddr.sun_family = AF_UNIX;
        strncpy(sockaddr.sun_path, sock_name.c_str(), sizeof(sockaddr.sun_path) - 1);

        sock_err = bind(sock, (struct sockaddr *)&sockaddr, sockaddr_len);
        CCL_THROW_IF_NOT(sock_err != ccl::utils::invalid_err_code,
                         "bind failed: sock_err: ",
                         sock_err,
                         ", errno: ",
                         strerror(errno));

        // listen to the socket to accept a connection to the other process
        sock_err = listen(sock, comm_size);
        CCL_THROW_IF_NOT(sock_err != ccl::utils::invalid_err_code,
                         "listen failed: sock_err: ",
                         sock_err,
                         ", errno: ",
                         strerror(errno));

        // notify the other process that the socket is created and being listened to
        ccl::utils::allgather(comm, &ack, all_ints.data(), sizeof(int));

        all_socks[0] = accept(sock, (struct sockaddr *)&sockaddr, &sockaddr_len);
        CCL_THROW_IF_NOT(all_socks[0] != ccl::utils::invalid_err_code,
                         "accept failed: sock: ",
                         all_socks[0],
                         ", errno: ",
                         strerror(errno));

        if (setsockopt(all_socks[0], SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) ==
            ccl::utils::invalid_err_code) {
            CCL_THROW("setsockopt failed: sock: ", all_socks[0], ", errno: ", strerror(errno));
        }
        ccl::utils::close_fd(sock);
    }

    LOG_DEBUG("connection is set up");
    device_fds = setup_device_fds(comm_size, rank_idx, device_bdfs);
    physical_devices = fill_physical_devices();

    // close sockets
    if (rank_idx == 0) {
        close_sockets(comm_size, rank_idx);
        ccl::utils::allgather(comm, &ack, all_ints.data(), sizeof(int));
    }
    else {
        ccl::utils::allgather(comm, &ack, all_ints.data(), sizeof(int));
        close_sockets(comm_size, rank_idx);
    }
}

void fd_manager::close_sockets(int comm_size, int rank_idx) {
    int sock_err;
    std::string sock_name;
    for (int i = 0; i < comm_size; ++i) {
        if (all_socks[i] != ccl::utils::invalid_fd) {
            sock_err = close(all_socks[i]);
            CCL_THROW_IF_NOT(sock_err != ccl::utils::invalid_err_code,
                             "close failed: ret",
                             sock_err,
                             ", errno: ",
                             strerror(errno));
        }

        if (all_pids[rank_idx] != ccl::utils::invalid_pid && rank_idx != i) {
            sock_name = "/tmp/ccl-ipc-fd-sock-" + std::to_string(all_pids[rank_idx]) + ":" +
                        std::to_string(rank_idx) + "-" + std::to_string(i) + "-" +
                        std::to_string(comm_size);
            unlink(sock_name.c_str());
        }
    }
}

} // namespace ze
} // namespace ccl
