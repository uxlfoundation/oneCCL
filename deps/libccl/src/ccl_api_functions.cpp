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

#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/environment.hpp"
#include "oneapi/ccl/api_functions.hpp"
#include "comm/comm.hpp"
#include "oneapi/ccl/exception.hpp"
#include "coll/group/group.hpp"
#include "comm/windows.hpp"
#include "common/utils/device_handle_map.hpp"
#include "common/window/window_impl.hpp"
#include "oneapi/ccl/device_api/window_data.hpp"

#include <type_traits>
#include <vector>

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "comm/comm_interface.hpp"
#endif //#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

#include "ccl_api_functions_generators.hpp"
#include "common/global/global.hpp"
#include "common/api_wrapper/mpi_api_wrapper.hpp"
// WA for broadcast to avoid scheduler completely
#include "coll/algorithms/broadcast/mpi_bcast_invoke.hpp"
namespace ccl {

namespace v1 {

/**
 * A structure that is a friend of the passed object
 * and which allows access to the internal representation of this object
 */
struct impl_dispatch {
    template <class Object>
    const typename Object::impl_value_t& operator()(const Object& obj) {
        return obj.get_impl();
    }
};

void init(const init_attr& attr) {
    auto& env = detail::environment::instance();
    (void)env;
}

/******************** ENVIRONMENT ********************/

library_version get_library_version() {
    return detail::environment::get_library_version();
}

/* datatype */
datatype register_datatype(const datatype_attr& attr) {
    return detail::environment::instance().register_datatype(attr);
}

void deregister_datatype(datatype dtype) {
    return detail::environment::instance().deregister_datatype(dtype);
}

size_t get_datatype_size(datatype dtype) {
    return detail::environment::instance().get_datatype_size(dtype);
}

/* KVS */
shared_ptr_class<kvs> create_main_kvs(const kvs_attr& attr) {
    return detail::environment::instance().create_main_kvs(attr);
}

shared_ptr_class<kvs> create_kvs(const kvs::address_type& addr, const kvs_attr& attr) {
    return detail::environment::instance().create_kvs(addr, attr);
}

/* device */
device create_device() {
    static empty_t empty{};
    return detail::environment::instance().create_device(empty);
}

/* context */
context create_context() {
    static empty_t empty{};
    return detail::environment::instance().create_context(empty);
}

/* stream */
stream create_stream() {
    return default_stream;
}

} // namespace v1

namespace preview {

/* communicator */
communicator create_communicator(const comm_attr& attr) {
    return ccl::detail::environment::instance().create_communicator(attr);
}

communicator create_communicator(const int size,
                                 shared_ptr_class<kvs_interface> kvs,
                                 const comm_attr& attr) {
    return ccl::detail::environment::instance().create_communicator(size, kvs, attr);
}

} // namespace preview

namespace v1 {
/***************** SPLIT COMMUNICATOR *****************/
communicator split_communicator(const communicator& comm, int color, int key) {
    return ccl::detail::environment::instance().split_communicator(comm, color, key);
}

/******************** COMMUNICATOR ********************/
communicator create_communicator(const int size,
                                 const int rank,
                                 shared_ptr_class<kvs_interface> kvs,
                                 const comm_attr& attr) {
    return detail::environment::instance().create_communicator(size, rank, kvs, attr);
}

/*************** CUSTOM REDUCTIONS ****************/

void reduction_create_pre_mul_sum(reduction* rtype,
                                  void* scalar,
                                  datatype dtype,
                                  scalar_residence_type residence,
                                  const communicator& comm) {
    ccl::detail::environment::instance().reduction_create_pre_mul_sum(
        rtype, scalar, dtype, residence, comm);
}

void reduction_destroy(reduction rtype, const communicator& comm) {
    ccl::detail::environment::instance().reduction_destroy(rtype, comm);
}

/******************** GROUP CALLS ********************/

void group_start() {
    group_impl::start();
}

void group_end() {
    group_impl::end();
}

/******************** MEMORY CALLS ********************/

void CCL_API mem_alloc(const stream& op_stream, size_t size, void** ptr) {
#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
    impl_dispatch disp;
    auto global_stream = disp(op_stream);
    sycl::queue q = global_stream->get_native_stream();
    *ptr = sycl::malloc_device<char>(size, q);
#else
    *ptr = nullptr;
#endif
}

void CCL_API mem_free(const stream& op_stream, void* ptr) {
#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
    impl_dispatch disp;
    auto global_stream = disp(op_stream);
    sycl::queue q = global_stream->get_native_stream();
    sycl::free(ptr, q);
#endif
}

/******************** BUFFER REGISTRATION CALLS ********************/

void comm_register(const communicator& comm, void* buffer, size_t size, void** handle) {
    LOG_DEBUG("registering buffer: ", buffer, ", size: ", size);

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
    impl_dispatch disp;
    // Get the underlying comm implementation to access the ccl_comm
    auto& comm_impl = disp(comm);

    static_cast<ccl_comm*>(comm_impl.get())->buffer_register(buffer, size, handle);
#else
    *handle = NULL;
#endif
}

void comm_deregister(const communicator& comm, void* handle) {
    LOG_DEBUG("deregistering buffer. handle:", handle);

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
    impl_dispatch disp;
    // Get the underlying comm implementation to access the ccl_comm
    auto& comm_impl = disp(comm);

    static_cast<ccl_comm*>(comm_impl.get())->buffer_deregister(handle);
#endif
}

/******************** WINDOW REGISTRATION CALLS ********************/

namespace {

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)

// device API only support ARC cards
bool device_api_supported() {
    const auto& ze_data = ccl::global_data::get().ze_data;
    return ze_data && !ze_data->devices.empty() && ze::is_arc_card(ze_data->devices[0].family);
}

/** Host state of a registered window; see utils::device_handle_map. */
struct window_record {
    /** Mirror of what was published to the device; every pointer in it is a device address. */
    window_data host_copy{};
    std::vector<void*> host_ptrs;          /**< peer base pointers, for the host resolves */
    ccl_window* internal_window = nullptr; /**< host-only: IPC handle lifetime */
};

static_assert(std::is_trivially_copyable<window_data>::value,
              "window_data is published to the device with a plain memcpy");

utils::device_handle_map<window_record> window_records;

#endif // defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)

} // anonymous namespace

window_t window_register_internal(ccl_comm* ccl_comm_ptr,
                                  void* buffer,
                                  size_t size,
                                  int win_flags) {
    LOG_DEBUG("registering window, buffer: ", buffer, ", size: ", size);

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (!device_api_supported()) {
        LOG_WARN("Window registration is not supported on this device");
        return nullptr;
    }

    sycl::device& dev = ccl_comm_ptr->get_device()->get_native();
    sycl::context& ctx = ccl_comm_ptr->get_context()->get_native();
    // The LSA team is the node communicator
    auto node_comm = ccl_comm_ptr->get_node_comm();

    // Fill the host record, then publish it to the device in one copy.
    window_record record;
    window_data& host_copy = record.host_copy;
    host_copy.ptrs = nullptr;
    host_copy.nranks = 0;
    host_copy.size = size;
    host_copy.world_rank = ccl_comm_ptr->rank();
    host_copy.lsa_size = static_cast<int>(node_comm->size());
    host_copy.lsa_rank = node_comm->rank();

    if (win_flags == CCL_WIN_COLL_SYMMETRIC) {
        // The get_*_pointer world->LSA conversion assumes this node's world ranks
        // are consecutive (node rank i -> comm rank world_base + i). Checked before
        // registering anything, so an unsupported layout fails loudly here rather
        // than silently reading the wrong peer in a kernel.
        int world_base = host_copy.world_rank - host_copy.lsa_rank;
        for (int i = 0; i < host_copy.lsa_size; i++) {
            int comm_peer = ccl_comm_ptr->get_rank_from_global(node_comm->get_global_rank(i));
            CCL_THROW_IF_NOT(
                comm_peer == world_base + i,
                "window registration: node ranks are not contiguous in "
                "communicator order (node rank ",
                i,
                " -> comm rank ",
                comm_peer,
                ", expected ",
                world_base + i,
                "); this rank layout is not supported by the device API");
        }

        record.internal_window = node_comm->window_register(buffer, size);
        if (record.internal_window) {
            record.host_ptrs = record.internal_window->get_ptrs(0);
            host_copy.nranks = static_cast<int>(record.host_ptrs.size());
        }
    }
    else {
        LOG_WARN("Window registration is not support, CCL_WIN_COLL_SYMMETRIC ignored");
    }

    // The pointer array kernels index, and the window struct itself.
    if (host_copy.nranks > 0) {
        host_copy.ptrs = sycl::malloc_device<void*>(host_copy.nranks, dev, ctx);
    }
    window_data* dev_win = sycl::malloc_device<window_data>(1, dev, ctx);

    sycl::queue q(ctx, dev);
    if (host_copy.ptrs) {
        q.memcpy(host_copy.ptrs, record.host_ptrs.data(), host_copy.nranks * sizeof(void*));
    }
    q.memcpy(dev_win, &host_copy, sizeof(window_data));
    q.wait();

    window_records.insert(dev_win, std::move(record));
    return dev_win;
#else
    CCL_THROW("Window registration requires SYCL and Level Zero");
    return nullptr;
#endif
}

void window_deregister_internal(ccl_comm* ccl_comm_ptr, window_t& win) {
    LOG_DEBUG("deregistering window");

    if (!win) return;

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    // win addresses device memory — do not dereference it; the record has everything.
    window_record record = window_records.remove("comm_window_deregister", win);

    sycl::context& ctx = ccl_comm_ptr->get_context()->get_native();

    // Deregister the internal window (closes IPC handles)
    if (record.internal_window) {
        auto node_comm = ccl_comm_ptr->get_node_comm();
        node_comm->window_deregister(record.internal_window);
    }

    if (record.host_copy.ptrs) {
        sycl::free(record.host_copy.ptrs, ctx);
    }
    sycl::free(win, ctx);
#else
    CCL_THROW("Window deregistration requires SYCL and Level Zero");
#endif

    win = nullptr;
}

namespace {

ccl_comm* to_ccl_comm(const communicator& comm) {
    impl_dispatch disp;
    auto& comm_impl = disp(comm);
    return static_cast<ccl_comm*>(comm_impl.get());
}

} // anonymous namespace

/* device API */

window_t CCL_API comm_window_register_v2(const communicator& comm,
                                         void* buffer,
                                         size_t size,
                                         int win_flags) {
    return window_register_internal(to_ccl_comm(comm), buffer, size, win_flags);
}

void CCL_API comm_window_deregister(const communicator& comm, window_t& win) {
    window_deregister_internal(to_ccl_comm(comm), win);
}

/* v1 ABI: window is an RAII box around the same registration */

window CCL_API comm_window_register(const communicator& comm,
                                    void* buffer,
                                    size_t size,
                                    int win_flags) {
    auto* ccl_comm_ptr = to_ccl_comm(comm);
    // A null win still yields a live window, as in 2022.1.0.
    window_t win = window_register_internal(ccl_comm_ptr, buffer, size, win_flags);
    return window(std::make_shared<ccl_window_impl>(ccl_comm_ptr, win));
}

void CCL_API comm_window_deregister(const communicator& comm, window& win) {
    // The release happens in ~ccl_window_impl; comm is unused, as in 2022.1.0.
    (void)comm;
    win = window(window::impl_value_t(nullptr));
}

void get_peer_device_pointer(window_t win, size_t offset, int peer, void** out_ptr) {
    if (!out_ptr) {
        CCL_THROW("get_peer_device_pointer: out_ptr must not be null");
    }
    if (!win) {
        *out_ptr = nullptr;
        return;
    }
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    const window_record* record = nullptr;
    auto lock = window_records.at("get_peer_device_pointer", win, &record);
    const std::vector<void*>& ptrs = record->host_ptrs;
    // Convert the world rank to an LSA slot, ptrs[] being LSA-rank indexed
    // (node ranks are contiguous in world order, enforced at registration).
    int slot = record->host_copy.lsa_rank + (peer - record->host_copy.world_rank);
    if (offset >= record->host_copy.size || slot < 0 ||
        slot >= record->host_copy.lsa_size || slot >= static_cast<int>(ptrs.size()) ||
        !ptrs[slot]) {
        *out_ptr = nullptr;
        return;
    }
    *out_ptr = static_cast<char*>(ptrs[slot]) + offset;
    LOG_DEBUG("get_peer_device_pointer: peer=", peer,
              " slot=", slot,
              " base=", ptrs[slot],
              " offset=", offset,
              " result=", *out_ptr);
#else
    CCL_THROW("get_peer_device_pointer requires SYCL and Level Zero");
#endif
}

void get_lsa_device_pointer(window_t win, size_t offset, int lsa_peer, void** out_ptr) {
    if (!out_ptr) {
        CCL_THROW("get_lsa_device_pointer: out_ptr must not be null");
    }
    if (!win) {
        *out_ptr = nullptr;
        return;
    }
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    const window_record* record = nullptr;
    auto lock = window_records.at("get_lsa_device_pointer", win, &record);
    const std::vector<void*>& ptrs = record->host_ptrs;
    // ptrs[] is LSA-rank indexed, so lsa_peer indexes it directly.
    if (offset >= record->host_copy.size || lsa_peer < 0 ||
        lsa_peer >= record->host_copy.lsa_size ||
        lsa_peer >= static_cast<int>(ptrs.size()) || !ptrs[lsa_peer]) {
        *out_ptr = nullptr;
        return;
    }
    *out_ptr = static_cast<char*>(ptrs[lsa_peer]) + offset;
    LOG_DEBUG("get_lsa_device_pointer: lsa_peer=", lsa_peer,
              " base=", ptrs[lsa_peer],
              " offset=", offset,
              " result=", *out_ptr);
#else
    CCL_THROW("get_lsa_device_pointer requires SYCL and Level Zero");
#endif
}

void get_local_device_pointer(window_t win, size_t offset, void** out_ptr) {
    if (!out_ptr) {
        CCL_THROW("get_local_device_pointer: out_ptr must not be null");
    }
    if (!win) {
        *out_ptr = nullptr;
        return;
    }
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    const window_record* record = nullptr;
    auto lock = window_records.at("get_local_device_pointer", win, &record);
    const std::vector<void*>& ptrs = record->host_ptrs;
    int slot = record->host_copy.lsa_rank;
    if (offset >= record->host_copy.size || slot < 0 ||
        slot >= record->host_copy.lsa_size || slot >= static_cast<int>(ptrs.size()) ||
        !ptrs[slot]) {
        *out_ptr = nullptr;
        return;
    }
    *out_ptr = static_cast<char*>(ptrs[slot]) + offset;
    LOG_DEBUG("get_local_device_pointer:"
              " base=", ptrs[slot],
              " offset=", offset,
              " result=", *out_ptr);
#else
    CCL_THROW("get_local_device_pointer requires SYCL and Level Zero");
#endif
}

/******************** DEVICE COMMUNICATOR ********************/

} // namespace v1
} // namespace ccl

// Forward declaration (defined in simple_ipc.cpp at global scope)
std::vector<std::vector<void*>> do_simple_ipc_exchange(ccl_comm* comm,
                                                       const std::vector<void*>& ptrs,
                                                       const std::vector<size_t>& sizes);

namespace ccl {
namespace v1 {

namespace {

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)

/**
 * The communicator is a value the caller owns, so only the peer inbox pointers
 * need a host record: teardown closes an IPC handle per peer. Keyed by the
 * device array kernels index.
 */
struct dev_comm_record {
    int rank = 0;                        /**< index of this rank's own inbox */
    std::vector<uint32_t*> host_inboxes; /**< peer inbox bases, for IPC close at teardown */
};

utils::device_handle_map<dev_comm_record> dev_comm_records;

#endif // defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)

} // anonymous namespace

void dev_comm_create(const communicator& comm,
                     const dev_comm_requirements* reqs,
                     dev_comm_t* out_dev_comm) {
    LOG_DEBUG("creating device communicator, barrier_count: ", reqs->lsa_barrier_count);

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (!device_api_supported()) {
        LOG_WARN("Device communicator is not supported on this device");
        // Leave the caller holding an empty communicator: barrier_inboxes stays
        // null, which is what dev_comm_destroy and the callers test for.
        *out_dev_comm = dev_comm_t{};
        return;
    }

    impl_dispatch disp;
    auto& comm_impl = disp(comm);
    auto* ccl_comm_ptr = static_cast<ccl_comm*>(comm_impl.get());
    sycl::device& dev = ccl_comm_ptr->get_device()->get_native();
    sycl::context& ctx = ccl_comm_ptr->get_context()->get_native();

    int rank = ccl_comm_ptr->rank();
    int nranks = ccl_comm_ptr->size();
    int n_barriers = reqs->lsa_barrier_count;

    // Local barrier inbox in device memory, written by peers over IPC:
    // n_barriers * nranks slots, indexed [barrier_idx * nranks + sender_rank].
    uint32_t* local_inbox = sycl::malloc_device<uint32_t>(n_barriers * nranks, dev, ctx);
    sycl::queue q(ctx, dev);
    q.memset(local_inbox, 0, n_barriers * nranks * sizeof(uint32_t)).wait();

    // Exchange inbox pointers across all ranks via IPC
    std::vector<void*> in_buffers = { static_cast<void*>(local_inbox) };
    std::vector<size_t> in_sizes = { static_cast<size_t>(n_barriers * nranks) * sizeof(uint32_t) };

    auto node_comm = ccl_comm_ptr->get_node_comm();
    auto* node_comm_raw = node_comm.get();
    std::vector<std::vector<void*>> ipc_ptrs =
        ::do_simple_ipc_exchange(node_comm_raw, in_buffers, in_sizes);

    // Fill the host record, then publish the inbox array to the device.
    dev_comm_record record;
    record.rank = rank;
    record.host_inboxes.resize(nranks, nullptr);
    for (int r = 0; r < nranks; r++) {
        record.host_inboxes[r] =
            (r == rank) ? local_inbox : static_cast<uint32_t*>(ipc_ptrs[r][0]);
    }

    // Fill in the caller's communicator; kernels capture it by value.
    out_dev_comm->rank = rank;
    out_dev_comm->nranks = nranks;
    // A node's ranks, consecutive in world rank order and uniform across nodes
    // (lsa_size = gcd of all node sizes).
    out_dev_comm->lsa_size = static_cast<int>(node_comm_raw->size());
    out_dev_comm->lsa_rank = rank % out_dev_comm->lsa_size;
    out_dev_comm->n_barriers = n_barriers;

    out_dev_comm->barrier_inboxes = sycl::malloc_device<uint32_t*>(nranks, dev, ctx);
    out_dev_comm->barrier_epochs = sycl::malloc_device<uint32_t>(n_barriers, dev, ctx);

    q.memcpy(out_dev_comm->barrier_inboxes, record.host_inboxes.data(),
             nranks * sizeof(uint32_t*));
    q.memset(out_dev_comm->barrier_epochs, 0, n_barriers * sizeof(uint32_t));
    q.wait();

    dev_comm_records.insert(out_dev_comm->barrier_inboxes, std::move(record));

    LOG_DEBUG("device communicator created: rank=", rank, " nranks=", nranks,
              " n_barriers=", n_barriers);
#else
    CCL_THROW("Device communicator requires SYCL and Level Zero");
#endif
}

void dev_comm_destroy(const communicator& comm, const dev_comm_t* dev_comm) {
    LOG_DEBUG("destroying device communicator");

    // A zero-initialized communicator was never created: nothing to free.
    if (!dev_comm || !dev_comm->barrier_inboxes) return;

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    dev_comm_record record =
        dev_comm_records.remove("dev_comm_destroy", dev_comm->barrier_inboxes);

    impl_dispatch disp;
    auto& comm_impl = disp(comm);
    auto* ccl_comm_ptr = static_cast<ccl_comm*>(comm_impl.get());
    sycl::context& ctx = ccl_comm_ptr->get_context()->get_native();

    // From the record, not the caller's struct, which may have been overwritten.
    int rank = record.rank;
    int nranks = static_cast<int>(record.host_inboxes.size());
    uint32_t* local_inbox = record.host_inboxes[rank];

    // Close IPC handles for remote inboxes
    ze_context_handle_t ze_ctx{};
    ze_device_handle_t ze_dev{};
    ze_memory_allocation_properties_t mem_props{};
    if (local_inbox &&
        ccl::ze::get_buffer_context_and_device(local_inbox, &ze_ctx, &ze_dev, &mem_props)) {
        for (int r = 0; r < nranks; r++) {
            if (r == rank) continue;
            if (record.host_inboxes[r]) {
                ZE_CALL(zeMemCloseIpcHandle, (ze_ctx, record.host_inboxes[r]));
            }
        }
    }

    // Free local inbox (device memory)
    if (local_inbox) {
        sycl::free(local_inbox, ctx);
    }

    // Free the inbox pointer array
    sycl::free(dev_comm->barrier_inboxes, ctx);

    // Free the barrier epoch array
    if (dev_comm->barrier_epochs) {
        sycl::free(dev_comm->barrier_epochs, ctx);
    }
    // Nothing to free for the communicator itself — it is the caller's value.
#else
    CCL_THROW("Device communicator requires SYCL and Level Zero");
#endif
}

/******************** OPERATION ********************/

#define CHECK_DEPS(deps) \
    do { \
        if (!deps.empty()) { \
            throw ccl::exception( \
                std::string(__PRETTY_FUNCTION__) + \
                " - handling a vector of events that the operation should depend on is not implemented"); \
        } \
    } while (0)

/* allgather */
event allgather(const void* send_buf,
                void* recv_buf,
                size_t count,
                datatype dtype,
                const communicator& comm,
                const stream& op_stream,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, dtype, disp(op_stream), attr, deps);
}

event allgather(const void* send_buf,
                void* recv_buf,
                size_t count,
                datatype dtype,
                const communicator& comm,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(
        send_buf, recv_buf, count, dtype, disp(default_stream), attr, deps);
}

event allgather(const void* send_buf,
                const vector_class<void*>& recv_buf,
                size_t count,
                datatype dtype,
                const communicator& comm,
                const stream& op_stream,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, dtype, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event allgather(const BufferType* send_buf,
                BufferType* recv_buf,
                size_t count,
                const communicator& comm,
                const stream& op_stream,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event allgather(const BufferType* send_buf,
                BufferType* recv_buf,
                size_t count,
                const communicator& comm,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event allgather(const BufferType* send_buf,
                vector_class<BufferType*>& recv_buf,
                size_t count,
                const communicator& comm,
                const stream& op_stream,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event allgather(const BufferType* send_buf,
                vector_class<BufferType*>& recv_buf,
                size_t count,
                const communicator& comm,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgather(const BufferObjectType& send_buf,
                BufferObjectType& recv_buf,
                size_t count,
                const communicator& comm,
                const stream& op_stream,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgather(const BufferObjectType& send_buf,
                BufferObjectType& recv_buf,
                size_t count,
                const communicator& comm,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgather(const BufferObjectType& send_buf,
                vector_class<reference_wrapper_class<BufferObjectType>>& recv_buf,
                size_t count,
                const communicator& comm,
                const stream& op_stream,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgather(const BufferObjectType& send_buf,
                vector_class<reference_wrapper_class<BufferObjectType>>& recv_buf,
                size_t count,
                const communicator& comm,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

/* allgatherv */
event allgatherv(const void* send_buf,
                 size_t send_count,
                 void* recv_buf,
                 const vector_class<size_t>& recv_counts,
                 datatype dtype,
                 const communicator& comm,
                 const stream& op_stream,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, dtype, disp(op_stream), attr, deps);
}

event allgatherv(const void* send_buf,
                 size_t send_count,
                 void* recv_buf,
                 const vector_class<size_t>& recv_counts,
                 datatype dtype,
                 const communicator& comm,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, dtype, disp(default_stream), attr, deps);
}

event allgatherv(const void* send_buf,
                 size_t send_count,
                 const vector_class<void*>& recv_bufs,
                 const vector_class<size_t>& recv_counts,
                 datatype dtype,
                 const communicator& comm,
                 const stream& op_stream,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, dtype, disp(op_stream), attr, deps);
}

event allgatherv(const void* send_buf,
                 size_t send_count,
                 const vector_class<void*>& recv_bufs,
                 const vector_class<size_t>& recv_counts,
                 datatype dtype,
                 const communicator& comm,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, dtype, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event allgatherv(const BufferType* send_buf,
                 size_t send_count,
                 BufferType* recv_buf,
                 const vector_class<size_t>& recv_counts,
                 const communicator& comm,
                 const stream& op_stream,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event allgatherv(const BufferType* send_buf,
                 size_t send_count,
                 BufferType* recv_buf,
                 const vector_class<size_t>& recv_counts,
                 const communicator& comm,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event allgatherv(const BufferType* send_buf,
                 size_t send_count,
                 vector_class<BufferType*>& recv_bufs,
                 const vector_class<size_t>& recv_counts,
                 const communicator& comm,
                 const stream& op_stream,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event allgatherv(const BufferType* send_buf,
                 size_t send_count,
                 vector_class<BufferType*>& recv_bufs,
                 const vector_class<size_t>& recv_counts,
                 const communicator& comm,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgatherv(const BufferObjectType& send_buf,
                 size_t send_count,
                 BufferObjectType& recv_buf,
                 const vector_class<size_t>& recv_counts,
                 const communicator& comm,
                 const stream& op_stream,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgatherv(const BufferObjectType& send_buf,
                 size_t send_count,
                 BufferObjectType& recv_buf,
                 const vector_class<size_t>& recv_counts,
                 const communicator& comm,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgatherv(const BufferObjectType& send_buf,
                 size_t send_count,
                 vector_class<ccl::reference_wrapper_class<BufferObjectType>>& recv_bufs,
                 const vector_class<size_t>& recv_counts,
                 const communicator& comm,
                 const stream& op_stream,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgatherv(const BufferObjectType& send_buf,
                 size_t send_count,
                 vector_class<ccl::reference_wrapper_class<BufferObjectType>>& recv_bufs,
                 const vector_class<size_t>& recv_counts,
                 const communicator& comm,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, disp(default_stream), attr, deps);
}

/* allreduce */
event allreduce(const void* send_buf,
                void* recv_buf,
                size_t count,
                datatype dtype,
                reduction reduction,
                const communicator& comm,
                const stream& op_stream,
                const allreduce_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allreduce(
        send_buf, recv_buf, count, dtype, reduction, disp(op_stream), attr, deps);
}

event allreduce(const void* send_buf,
                void* recv_buf,
                size_t count,
                datatype dtype,
                reduction reduction,
                const communicator& comm,
                const allreduce_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allreduce(
        send_buf, recv_buf, count, dtype, reduction, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event allreduce(const BufferType* send_buf,
                BufferType* recv_buf,
                size_t count,
                reduction reduction,
                const communicator& comm,
                const stream& op_stream,
                const allreduce_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allreduce(send_buf, recv_buf, count, reduction, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event allreduce(const BufferType* send_buf,
                BufferType* recv_buf,
                size_t count,
                reduction reduction,
                const communicator& comm,
                const allreduce_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allreduce(
        send_buf, recv_buf, count, reduction, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allreduce(const BufferObjectType& send_buf,
                BufferObjectType& recv_buf,
                size_t count,
                reduction reduction,
                const communicator& comm,
                const stream& op_stream,
                const allreduce_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allreduce(send_buf, recv_buf, count, reduction, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allreduce(const BufferObjectType& send_buf,
                BufferObjectType& recv_buf,
                size_t count,
                reduction reduction,
                const communicator& comm,
                const allreduce_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allreduce(
        send_buf, recv_buf, count, reduction, disp(default_stream), attr, deps);
}

/* alltoall */
event alltoall(const void* send_buf,
               void* recv_buf,
               size_t count,
               datatype dtype,
               const communicator& comm,
               const stream& op_stream,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, dtype, disp(op_stream), attr, deps);
}

event alltoall(const void* send_buf,
               void* recv_buf,
               size_t count,
               datatype dtype,
               const communicator& comm,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, dtype, disp(default_stream), attr, deps);
}

event alltoall(const vector_class<void*>& send_buf,
               const vector_class<void*>& recv_buf,
               size_t count,
               datatype dtype,
               const communicator& comm,
               const stream& op_stream,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, dtype, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoall(const BufferType* send_buf,
               BufferType* recv_buf,
               size_t count,
               const communicator& comm,
               const stream& op_stream,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoall(const BufferType* send_buf,
               BufferType* recv_buf,
               size_t count,
               const communicator& comm,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoall(const vector_class<BufferType*>& send_buf,
               const vector_class<BufferType*>& recv_buf,
               size_t count,
               const communicator& comm,
               const stream& op_stream,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoall(const vector_class<BufferType*>& send_buf,
               const vector_class<BufferType*>& recv_buf,
               size_t count,
               const communicator& comm,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoall(const BufferObjectType& send_buf,
               BufferObjectType& recv_buf,
               size_t count,
               const communicator& comm,
               const stream& op_stream,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoall(const BufferObjectType& send_buf,
               BufferObjectType& recv_buf,
               size_t count,
               const communicator& comm,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoall(const vector_class<reference_wrapper_class<BufferObjectType>>& send_buf,
               const vector_class<reference_wrapper_class<BufferObjectType>>& recv_buf,
               size_t count,
               const communicator& comm,
               const stream& op_stream,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoall(const vector_class<reference_wrapper_class<BufferObjectType>>& send_buf,
               const vector_class<reference_wrapper_class<BufferObjectType>>& recv_buf,
               size_t count,
               const communicator& comm,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

/* alltoallv */
event alltoallv(const void* send_buf,
                const vector_class<size_t>& send_counts,
                void* recv_buf,
                const vector_class<size_t>& recv_counts,
                datatype dtype,
                const communicator& comm,
                const stream& op_stream,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, dtype, disp(op_stream), attr, deps);
}

event alltoallv(const void* send_buf,
                const vector_class<size_t>& send_counts,
                void* recv_buf,
                const vector_class<size_t>& recv_counts,
                datatype dtype,
                const communicator& comm,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, dtype, disp(default_stream), attr, deps);
}

event alltoallv(const vector_class<void*>& send_bufs,
                const vector_class<size_t>& send_counts,
                const vector_class<void*>& recv_bufs,
                const vector_class<size_t>& recv_counts,
                datatype dtype,
                const communicator& comm,
                const stream& op_stream,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, dtype, disp(op_stream), attr, deps);
}

event alltoallv(const vector_class<void*>& send_bufs,
                const vector_class<size_t>& send_counts,
                const vector_class<void*>& recv_bufs,
                const vector_class<size_t>& recv_counts,
                datatype dtype,
                const communicator& comm,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, dtype, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoallv(const BufferType* send_buf,
                const vector_class<size_t>& send_counts,
                BufferType* recv_buf,
                const vector_class<size_t>& recv_counts,
                const communicator& comm,
                const stream& op_stream,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoallv(const BufferType* send_buf,
                const vector_class<size_t>& send_counts,
                BufferType* recv_buf,
                const vector_class<size_t>& recv_counts,
                const communicator& comm,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoallv(const vector_class<BufferType*>& send_bufs,
                const vector_class<size_t>& send_counts,
                const vector_class<BufferType*>& recv_bufs,
                const vector_class<size_t>& recv_counts,
                const communicator& comm,
                const stream& op_stream,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoallv(const vector_class<BufferType*>& send_bufs,
                const vector_class<size_t>& send_counts,
                const vector_class<BufferType*>& recv_bufs,
                const vector_class<size_t>& recv_counts,
                const communicator& comm,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoallv(const BufferObjectType& send_buf,
                const vector_class<size_t>& send_counts,
                BufferObjectType& recv_buf,
                const vector_class<size_t>& recv_counts,
                const communicator& comm,
                const stream& op_stream,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoallv(const BufferObjectType& send_buf,
                const vector_class<size_t>& send_counts,
                BufferObjectType& recv_buf,
                const vector_class<size_t>& recv_counts,
                const communicator& comm,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoallv(const vector_class<reference_wrapper_class<BufferObjectType>>& send_bufs,
                const vector_class<size_t>& send_counts,
                const vector_class<reference_wrapper_class<BufferObjectType>>& recv_bufs,
                const vector_class<size_t>& recv_counts,
                const communicator& comm,
                const stream& op_stream,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoallv(const vector_class<reference_wrapper_class<BufferObjectType>>& send_bufs,
                const vector_class<size_t>& send_counts,
                const vector_class<reference_wrapper_class<BufferObjectType>>& recv_bufs,
                const vector_class<size_t>& recv_counts,
                const communicator& comm,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, disp(default_stream), attr, deps);
}

/* barrier */
event barrier(const communicator& comm,
              const stream& op_stream,
              const barrier_attr& attr,
              const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->barrier(disp(op_stream), attr, deps);
}

event barrier(const communicator& comm, const barrier_attr& attr, const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->barrier(disp(default_stream), attr, deps);
}

/* broadcast */
event broadcast(void* buf,
                size_t count,
                datatype dtype,
                int root,
                const communicator& comm,
                const stream& op_stream,
                const broadcast_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    if (ccl::global_data::env().use_mpi_bcast_wa) {
        ccl_comm* global_comm = (ccl_comm*)(disp(comm).get());
        return invoke_mpi_bcast(buf, count, dtype, root, global_comm, op_stream, attr, deps);
    }
    return disp(comm)->bcast(buf, count, dtype, root, disp(op_stream), attr, deps);
}

event broadcast(void* buf,
                size_t count,
                datatype dtype,
                int root,
                const communicator& comm,
                const broadcast_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->bcast(buf, count, dtype, root, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event broadcast(BufferType* buf,
                size_t count,
                int root,
                const communicator& comm,
                const stream& op_stream,
                const broadcast_attr& attr,
                const vector_class<event>& deps)

{
    impl_dispatch disp;
    return disp(comm)->bcast(buf, count, root, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event broadcast(BufferType* buf,
                size_t count,
                int root,
                const communicator& comm,
                const broadcast_attr& attr,
                const vector_class<event>& deps)

{
    impl_dispatch disp;
    return disp(comm)->bcast(buf, count, root, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event broadcast(BufferObjectType& buf,
                size_t count,
                int root,
                const communicator& comm,
                const stream& op_stream,
                const broadcast_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->bcast(buf, count, root, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event broadcast(BufferObjectType& buf,
                size_t count,
                int root,
                const communicator& comm,
                const broadcast_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->bcast(buf, count, root, disp(default_stream), attr, deps);
}

/* broadcast */
event broadcast(void* send_buf,
                void* recv_buf,
                size_t count,
                datatype dtype,
                int root,
                const communicator& comm,
                const stream& op_stream,
                const broadcast_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->broadcast(
        send_buf, recv_buf, count, dtype, root, disp(op_stream), attr, deps);
}

event broadcast(void* send_buf,
                void* recv_buf,
                size_t count,
                datatype dtype,
                int root,
                const communicator& comm,
                const broadcast_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->broadcast(
        send_buf, recv_buf, count, dtype, root, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event broadcast(BufferType* send_buf,
                BufferType* recv_buf,
                size_t count,
                int root,
                const communicator& comm,
                const stream& op_stream,
                const broadcast_attr& attr,
                const vector_class<event>& deps)

{
    impl_dispatch disp;
    return disp(comm)->broadcast(send_buf, recv_buf, count, root, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event broadcast(BufferType* send_buf,
                BufferType* recv_buf,
                size_t count,
                int root,
                const communicator& comm,
                const broadcast_attr& attr,
                const vector_class<event>& deps)

{
    impl_dispatch disp;
    return disp(comm)->broadcast(send_buf, recv_buf, count, root, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event broadcast(BufferObjectType& send_buf,
                BufferObjectType& recv_buf,
                size_t count,
                int root,
                const communicator& comm,
                const stream& op_stream,
                const broadcast_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->broadcast(send_buf, recv_buf, count, root, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event broadcast(BufferObjectType& send_buf,
                BufferObjectType& recv_buf,
                size_t count,
                int root,
                const communicator& comm,
                const broadcast_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->broadcast(send_buf, recv_buf, count, root, disp(default_stream), attr, deps);
}

/* reduce */
event reduce(const void* send_buf,
             void* recv_buf,
             size_t count,
             datatype dtype,
             reduction reduction,
             int root,
             const communicator& comm,
             const stream& op_stream,
             const reduce_attr& attr,
             const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, dtype, reduction, root, disp(op_stream), attr, deps);
}

event reduce(const void* send_buf,
             void* recv_buf,
             size_t count,
             datatype dtype,
             reduction reduction,
             int root,
             const communicator& comm,
             const reduce_attr& attr,
             const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, dtype, reduction, root, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event reduce(const BufferType* send_buf,
             BufferType* recv_buf,
             size_t count,
             reduction reduction,
             int root,
             const communicator& comm,
             const stream& op_stream,
             const reduce_attr& attr,
             const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, reduction, root, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event reduce(const BufferType* send_buf,
             BufferType* recv_buf,
             size_t count,
             reduction reduction,
             int root,
             const communicator& comm,
             const reduce_attr& attr,
             const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, reduction, root, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event reduce(const BufferObjectType& send_buf,
             BufferObjectType& recv_buf,
             size_t count,
             reduction reduction,
             int root,
             const communicator& comm,
             const stream& op_stream,
             const reduce_attr& attr,
             const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, reduction, root, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event reduce(const BufferObjectType& send_buf,
             BufferObjectType& recv_buf,
             size_t count,
             reduction reduction,
             int root,
             const communicator& comm,
             const reduce_attr& attr,
             const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, reduction, root, disp(default_stream), attr, deps);
}

/* reduce_scatter */
event reduce_scatter(const void* send_buf,
                     void* recv_buf,
                     size_t recv_count,
                     datatype dtype,
                     reduction reduction,
                     const communicator& comm,
                     const stream& op_stream,
                     const reduce_scatter_attr& attr,
                     const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, dtype, reduction, disp(op_stream), attr, deps);
}

event reduce_scatter(const void* send_buf,
                     void* recv_buf,
                     size_t recv_count,
                     datatype dtype,
                     reduction reduction,
                     const communicator& comm,
                     const reduce_scatter_attr& attr,
                     const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, dtype, reduction, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event reduce_scatter(const BufferType* send_buf,
                     BufferType* recv_buf,
                     size_t recv_count,
                     reduction reduction,
                     const communicator& comm,
                     const stream& op_stream,
                     const reduce_scatter_attr& attr,
                     const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, reduction, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event reduce_scatter(const BufferType* send_buf,
                     BufferType* recv_buf,
                     size_t recv_count,
                     reduction reduction,
                     const communicator& comm,
                     const reduce_scatter_attr& attr,
                     const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, reduction, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event reduce_scatter(const BufferObjectType& send_buf,
                     BufferObjectType& recv_buf,
                     size_t recv_count,
                     reduction reduction,
                     const communicator& comm,
                     const stream& op_stream,
                     const reduce_scatter_attr& attr,
                     const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, reduction, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event reduce_scatter(const BufferObjectType& send_buf,
                     BufferObjectType& recv_buf,
                     size_t recv_count,
                     reduction reduction,
                     const communicator& comm,
                     const reduce_scatter_attr& attr,
                     const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, reduction, disp(default_stream), attr, deps);
}

/* gather */
event gather(const void* send_buf,
             void* recv_buf,
             size_t count,
             datatype dtype,
             int root,
             const communicator& comm,
             const stream& op_stream,
             const gather_attr& attr,
             const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->gather(send_buf, recv_buf, count, dtype, root, disp(op_stream), attr, deps);
}

event gather(const void* send_buf,
             void* recv_buf,
             size_t count,
             datatype dtype,
             int root,
             const communicator& comm,
             const gather_attr& attr,
             const vector_class<event>& deps) {
    return gather(send_buf, recv_buf, count, dtype, root, comm, default_stream, attr, deps);
}

template <class BufferType, typename T>
event gather(const BufferType* send_buf,
             BufferType* recv_buf,
             size_t count,
             int root,
             const communicator& comm,
             const stream& op_stream,
             const gather_attr& attr,
             const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->gather(send_buf, recv_buf, count, root, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event gather(const BufferType* send_buf,
             BufferType* recv_buf,
             size_t count,
             int root,
             const communicator& comm,
             const gather_attr& attr,
             const vector_class<event>& deps) {
    return gather(send_buf, recv_buf, count, root, comm, default_stream, attr, deps);
}

template <class BufferObjectType, typename T>
event gather(const BufferObjectType& send_buf,
             BufferObjectType& recv_buf,
             size_t count,
             int root,
             const communicator& comm,
             const stream& op_stream,
             const gather_attr& attr,
             const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->gather(send_buf, recv_buf, count, root, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event gather(const BufferObjectType& send_buf,
             BufferObjectType& recv_buf,
             size_t count,
             int root,
             const communicator& comm,
             const gather_attr& attr,
             const vector_class<event>& deps) {
    return gather(send_buf, recv_buf, count, root, comm, default_stream, attr, deps);
}

/* scatter */
event scatter(const void* send_buf,
              void* recv_buf,
              size_t count,
              datatype dtype,
              int root,
              const communicator& comm,
              const stream& op_stream,
              const scatter_attr& attr,
              const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->scatter(send_buf, recv_buf, count, dtype, root, disp(op_stream), attr, deps);
}

event scatter(const void* send_buf,
              void* recv_buf,
              size_t count,
              datatype dtype,
              int root,
              const communicator& comm,
              const scatter_attr& attr,
              const vector_class<event>& deps) {
    return scatter(send_buf, recv_buf, count, dtype, root, comm, default_stream, attr, deps);
}

template <class BufferType, typename T>
event scatter(const BufferType* send_buf,
              BufferType* recv_buf,
              size_t count,
              int root,
              const communicator& comm,
              const stream& op_stream,
              const scatter_attr& attr,
              const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->scatter(send_buf, recv_buf, count, root, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event scatter(const BufferType* send_buf,
              BufferType* recv_buf,
              size_t count,
              int root,
              const communicator& comm,
              const scatter_attr& attr,
              const vector_class<event>& deps) {
    return scatter(send_buf, recv_buf, count, root, comm, default_stream, attr, deps);
}

template <class BufferObjectType, typename T>
event scatter(const BufferObjectType& send_buf,
              BufferObjectType& recv_buf,
              size_t count,
              int root,
              const communicator& comm,
              const stream& op_stream,
              const scatter_attr& attr,
              const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->scatter(send_buf, recv_buf, count, root, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event scatter(const BufferObjectType& send_buf,
              BufferObjectType& recv_buf,
              size_t count,
              int root,
              const communicator& comm,
              const scatter_attr& attr,
              const vector_class<event>& deps) {
    return scatter(send_buf, recv_buf, count, root, comm, default_stream, attr, deps);
}

/* recv */
event recv(void* recv_buf,
           size_t recv_count,
           datatype dtype,
           int peer,
           const communicator& comm,
           const stream& op_stream,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->recv(recv_buf, recv_count, dtype, peer, disp(op_stream), attr, deps);
}

event recv(void* recv_buf,
           size_t recv_count,
           datatype dtype,
           int peer,
           const communicator& comm,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->recv(recv_buf, recv_count, dtype, peer, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event recv(BufferType* recv_buf,
           size_t recv_count,
           int peer,
           const communicator& comm,
           const stream& op_stream,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->recv(recv_buf, recv_count, peer, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event recv(BufferType* recv_buf,
           size_t recv_count,
           int peer,
           const communicator& comm,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->recv(recv_buf, recv_count, peer, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event recv(BufferObjectType& recv_buf,
           size_t recv_count,
           int peer,
           const communicator& comm,
           const stream& op_stream,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->recv(recv_buf, recv_count, peer, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event recv(BufferObjectType& recv_buf,
           size_t recv_count,
           int peer,
           const communicator& comm,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->recv(recv_buf, recv_count, peer, disp(default_stream), attr, deps);
}

/* send */
event send(void* send_buf,
           size_t send_count,
           datatype dtype,
           int peer,
           const communicator& comm,
           const stream& op_stream,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->send(send_buf, send_count, dtype, peer, disp(op_stream), attr, deps);
}

event send(void* send_buf,
           size_t send_count,
           datatype dtype,
           int peer,
           const communicator& comm,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->send(send_buf, send_count, dtype, peer, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event send(BufferType* send_buf,
           size_t send_count,
           int peer,
           const communicator& comm,
           const stream& op_stream,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->send(send_buf, send_count, peer, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event send(BufferType* send_buf,
           size_t send_count,
           int peer,
           const communicator& comm,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->send(send_buf, send_count, peer, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event send(BufferObjectType& send_buf,
           size_t send_count,
           int peer,
           const communicator& comm,
           const stream& op_stream,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->send(send_buf, send_count, peer, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event send(BufferObjectType& send_buf,
           size_t send_count,
           int peer,
           const communicator& comm,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->send(send_buf, send_count, peer, disp(default_stream), attr, deps);
}

} // namespace v1

namespace v1 {

// API force instantiations for Operations
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(int8_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(uint8_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(int16_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(uint16_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(int32_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(uint32_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(int64_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(uint64_t);
/*API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(ccl::float16);*/
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(float);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(double);
/*API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(ccl::bfloat16);*/

#ifdef CCL_ENABLE_SYCL
#ifndef COMMA
#define COMMA ,
#endif

API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<int8_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<uint8_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<int16_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<uint16_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<int32_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<uint32_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<int64_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<uint64_t COMMA 1>);
/*API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<ccl::float16 COMMA 1>);*/
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<float COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<double COMMA 1>);
/*API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<ccl::bfloat16 COMMA 1>);*/

#undef COMMA
#endif // CCL_ENABLE_SYCL

} // namespace v1

} // namespace ccl
