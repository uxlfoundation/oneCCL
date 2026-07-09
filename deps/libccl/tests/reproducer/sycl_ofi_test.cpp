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

#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <iostream>
#include <map>
#include <set>
#include <string>

#include <sycl/sycl.hpp>

using namespace std;
using namespace sycl;
using namespace sycl::access;

#define RET_SUCCESS 0
#define RET_FAILURE 1

#define DEFAULT_OOB_PORT 12345
#define MAX_NUM_CHANNELS 16
#define EP_NAME_MAX_LEN  1024
#define MIN_MSG_SIZE     (1)
#define MAX_MSG_SIZE     (1 << 22)
#define ALIGN            (1 << 12)
#define MSG_TAG          (0xFFFF0000FFFF0000ULL)

#define GETTID() syscall(SYS_gettid)

#define LOG(fmt, ...) \
    do { \
        printf("(%ld) " fmt "\n", GETTID(), ##__VA_ARGS__); \
        fflush(stdout); \
    } while (0)

#define ASSERT(cond, fmt, ...) \
    do { \
        if (!(cond)) { \
            printf("FAILED\n"); \
            fprintf(stderr, \
                    "(%ld): %s:%s:%d: ASSERT '%s' FAILED: " fmt "\n", \
                    GETTID(), \
                    __FILE__, \
                    __FUNCTION__, \
                    __LINE__, \
                    #cond, \
                    ##__VA_ARGS__); \
            fflush(stderr); \
            throw std::runtime_error("ASSERT FAILED"); \
        } \
    } while (0)

#define CHK_OFI_ERR(name, cond, err) \
    do { \
        if (cond) { \
            fprintf(stderr, "%s: %s\n", name, strerror(-(err))); \
            ASSERT(0, "OFI error"); \
        } \
    } while (0)

#define SEND_MSG(ep, buf, len, peer, context) \
    do { \
        int err; \
        do { \
            err = fi_tsend(ep, buf, len, NULL, peer, MSG_TAG, context); \
            if (err != -FI_EAGAIN) { \
                CHK_OFI_ERR("fi_tsend", (err < 0), err); \
                break; \
            } \
        } while (err == -FI_EAGAIN); \
    } while (0)

#define RECV_MSG(ep, buf, len, peer, context) \
    do { \
        int err; \
        err = fi_trecv(ep, buf, len, NULL, peer, MSG_TAG, 0x0ULL, context); \
        CHK_OFI_ERR("fi_trecv", (err < 0), err); \
    } while (0)

#define WAIT_CQ(cq, n) \
    do { \
        struct fi_cq_tagged_entry entry[n]; \
        int ret, completed = 0; \
        while (completed < n) { \
            ret = fi_cq_read(cq, entry, n); \
            if (ret == -FI_EAGAIN) \
                continue; \
            CHK_OFI_ERR("fi_cq_read", (ret < 0), ret); \
            completed += ret; \
        } \
    } while (0)

typedef enum { ALLOC_REGULAR, ALLOC_RUNTIME } alloc_type;

struct options {
    int alloc_type;
    int channel_count;
    int is_client;
    char* server_name;
    int server_port;
};
struct options opt = { .alloc_type = ALLOC_REGULAR, .channel_count = 1 };

struct ofi_channel {
    struct fid_ep* ep;
    struct fid_cq* cq;
    fi_addr_t peer_addr;
    struct fi_context sctxt;
    struct fi_context rctxt;
    char* sbuf;
    char* rbuf;
};
struct ofi_channel ch[MAX_NUM_CHANNELS];

struct fi_info* fi;
struct fid_fabric* fabric;
struct fid_domain* domain;
struct fid_av* av;

sycl::queue gpu_queue;

char local_ep_name[EP_NAME_MAX_LEN];
char remote_ep_name[EP_NAME_MAX_LEN];

int oob_data_fd = 0, oob_connect_fd = 0;
struct sockaddr_in oob_sock_addr;

void send_oob(void* data_p, size_t datalen);
void recv_oob(void* data_p, size_t datalen);
void sync_oob();
void finalize_oob();
void connect_oob();

void init_oob() {
    struct hostent* addr;
    struct protoent* proto;
    char* host;
    int sock_fd;

    bzero((char*)&oob_sock_addr, sizeof(oob_sock_addr));

    if ((sock_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        LOG("can't open stream socket! errno=%d", errno);
        exit(RET_FAILURE);
    }

    if (!(proto = getprotobyname("tcp"))) {
        LOG("protocol 'tcp' unknown!");
        exit(RET_FAILURE);
    }

    if (opt.is_client) {
        if (atoi(opt.server_name) > 0) {
            oob_sock_addr.sin_family = AF_INET;
            oob_sock_addr.sin_addr.s_addr = inet_addr(opt.server_name);
        }
        else {
            if ((addr = gethostbyname(opt.server_name)) == NULL) {
                LOG("invalid hostname '%s'", opt.server_name);
                exit(RET_FAILURE);
            }

            oob_sock_addr.sin_family = addr->h_addrtype;
            memmove((char*)&(oob_sock_addr.sin_addr.s_addr), addr->h_addr, addr->h_length);
        }
        oob_sock_addr.sin_port = opt.server_port;
    }
    else {
        oob_sock_addr.sin_family = AF_INET;
        oob_sock_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        oob_sock_addr.sin_port = DEFAULT_OOB_PORT;

        while (bind(sock_fd, (struct sockaddr*)&oob_sock_addr, sizeof(oob_sock_addr)) < 0) {
            oob_sock_addr.sin_port++;
        }
        LOG("server_port: %d, use it as parameter for client process", oob_sock_addr.sin_port);
    }

    if (opt.is_client) {
        oob_data_fd = sock_fd;
    }
    else {
        oob_connect_fd = sock_fd;
    }

    connect_oob();
}

void connect_oob() {
    if (opt.is_client) {
        if (connect(oob_data_fd, (struct sockaddr*)&oob_sock_addr, sizeof(oob_sock_addr)) < 0) {
            LOG("client: cannot connect! errno=%d", errno);
            exit(RET_FAILURE);
        }
    }
    else {
        listen(oob_connect_fd, 5);
        oob_data_fd = accept(oob_connect_fd, NULL, NULL);
        if (oob_data_fd < 0) {
            LOG("server: accept failed! errno=%d", errno);
            exit(RET_FAILURE);
        }
    }
}

void finalize_oob() {
    char quit[] = "quit";
    if (opt.is_client) {
        write(oob_data_fd, quit, 5);
        read(oob_data_fd, quit, 5);
        close(oob_data_fd);
    }
    else {
        read(oob_data_fd, quit, 5);
        write(oob_data_fd, quit, 5);
        close(oob_data_fd);
        close(oob_connect_fd);
    }
}

int recv_fully_oob(int fd, void* obuf, int len) {
    int bytes_left = len;
    char* buf = (char*)obuf;
    int bytes_read = 0;

    while (bytes_left > 0 && (bytes_read = read(fd, (void*)buf, bytes_left)) > 0) {
        bytes_left -= bytes_read;
        buf += bytes_read;
    }
    if (bytes_read <= 0)
        return bytes_read;

    return len;
}

void send_oob(void* data_p, size_t datalen) {
    if (write(oob_data_fd, (char*)data_p, datalen) < 0) {
        LOG("write failed in send_oob: errno=%d", errno);
        exit(RET_FAILURE);
    }
}

void recv_oob(void* data_p, size_t datalen) {
    int bytes_read = 0;
    bytes_read = recv_fully_oob(oob_data_fd, (void*)data_p, datalen);

    if (bytes_read < 0) {
        LOG("read failed in recv_oob: errno=%d", errno);
        exit(RET_FAILURE);
    }
    else if (bytes_read != datalen) {
        LOG("partial read in RecvAuxData of %d bytes", bytes_read);
        exit(RET_FAILURE);
    }
}

void sync_oob() {
    char s[] = "sync_me";
    char response[7];

    if (write(oob_data_fd, s, strlen(s)) < 0 ||
        recv_fully_oob(oob_data_fd, response, strlen(s)) < 0) {
        perror("error writing or reading synchronization string");
        exit(RET_FAILURE);
    }

    if (strncmp(s, response, strlen(s))) {
        LOG("synchronization string incorrect!");
        exit(RET_FAILURE);
    }
}

bool has_gpu() {
    vector<device> devices = device::get_devices();
    for (const auto& device : devices) {
        if (device.is_gpu()) {
            return true;
        }
    }
    return false;
}

sycl::queue create_gpu_queue() {
    auto exception_handler = [&](exception_list elist) {
        for (exception_ptr const& e : elist) {
            try {
                rethrow_exception(e);
            }
            catch (std::exception const& e) {
                std::cout << "failure\n";
            }
        }
    };

    ASSERT(has_gpu(), "can not find GPU device");

    sycl::device dev;
    sycl::queue q;

    try {
        dev = sycl::device(sycl::gpu_selector());
        q = sycl::queue(dev, exception_handler);
        LOG("opened GPU device: %s", dev.get_info<info::device::name>().c_str());
    }
    catch (std::exception& e) {
        cerr << e.what() << "\n";
    }

    return q;
}

double when(void) {
    struct timeval tv;
    static struct timeval tv0;
    static int first = 1;
    int err;

    err = gettimeofday(&tv, NULL);
    if (err) {
        perror("gettimeofday");
        return 0;
    }

    if (first) {
        tv0 = tv;
        first = 0;
    }
    return (double)(tv.tv_sec - tv0.tv_sec) * 1.0e6 + (double)(tv.tv_usec - tv0.tv_usec);
}

void print_options(void) {
    printf("alloc_type = %s\n", (opt.alloc_type == ALLOC_REGULAR) ? "regular" : "runtime");
    printf("channel_count = %d\n", opt.channel_count);
    printf("is_client = %d\n", opt.is_client);
    printf("server_name = %s\n", opt.server_name);
    printf("server_port = %d\n", opt.server_port);
}

void alloc_buffers() {
    for (int i = 0; i < opt.channel_count; i++) {
        if (opt.alloc_type == ALLOC_RUNTIME) {
            ch[i].sbuf = (char*)aligned_alloc_host(ALIGN, MAX_MSG_SIZE, gpu_queue);
            ch[i].rbuf = (char*)aligned_alloc_host(ALIGN, MAX_MSG_SIZE, gpu_queue);

            auto sbuf_ptr_type = sycl::get_pointer_type(ch[i].sbuf, gpu_queue.get_context());
            auto rbuf_ptr_type = sycl::get_pointer_type(ch[i].rbuf, gpu_queue.get_context());

            ASSERT(sbuf_ptr_type == usm::alloc::host, "unexpected sbuf_ptr_type %d", sbuf_ptr_type);
            ASSERT(rbuf_ptr_type == usm::alloc::host, "unexpected rbuf_ptr_type %d", rbuf_ptr_type);

            LOG("allocated runtime buffers for channel #%d", i);
        }
        else {
            if (posix_memalign((void**)&ch[i].sbuf, ALIGN, MAX_MSG_SIZE)) {
                fprintf(stderr, "no memory\n");
                exit(RET_FAILURE);
            }

            if (posix_memalign((void**)&ch[i].rbuf, ALIGN, MAX_MSG_SIZE)) {
                fprintf(stderr, "no memory\n");
                exit(RET_FAILURE);
            }

            LOG("allocated regular buffers for channel #%d", i);
        }

        memset(ch[i].sbuf, 'a' + i, MAX_MSG_SIZE);
        memset(ch[i].rbuf, 'o' + i, MAX_MSG_SIZE);
    }
}

void free_buffers() {
    for (int i = 0; i < opt.channel_count; i++) {
        if (opt.alloc_type == ALLOC_RUNTIME) {
            sycl::free(ch[i].sbuf, gpu_queue);
            sycl::free(ch[i].rbuf, gpu_queue);
        }
        else {
            free(ch[i].sbuf);
            free(ch[i].rbuf);
        }
    }
}

void fill_buffers(int size) {
    for (int i = 0; i < opt.channel_count; i++) {
        memset(ch[i].sbuf, 'a' + i, size);
        memset(ch[i].rbuf, 'o' + i, size);
    }
}

void check_buffers(int size) {
    for (int i = 0; i < opt.channel_count; i++) {
        for (int j = 0; j < size; j++) {
            char expected = 'a' + i;
            ASSERT(ch[i].sbuf[j] == expected,
                   "unexpected sbuf[%d][%d] = %c, expected %c\n",
                   i,
                   j,
                   ch[i].sbuf[j],
                   expected);
            ASSERT(ch[i].rbuf[j] == expected,
                   "unexpected rbuf[%d][%d] = %c, expected %c\n",
                   i,
                   j,
                   ch[i].rbuf[j],
                   expected);
        }
    }
}

void init_ofi() {
    struct fi_info* hints;
    struct fi_cq_attr cq_attr;
    struct fi_cntr_attr cntr_attr;
    struct fi_av_attr av_attr;
    int err;
    int version;

    hints = fi_allocinfo();
    CHK_OFI_ERR("fi_allocinfo", (!hints), -ENOMEM);

    memset(&cq_attr, 0, sizeof(cq_attr));
    memset(&cntr_attr, 0, sizeof(cntr_attr));
    memset(&av_attr, 0, sizeof(av_attr));

    hints->ep_attr->type = FI_EP_RDM;
    hints->caps = FI_MSG | FI_TAGGED;
    hints->mode = FI_CONTEXT;
    hints->fabric_attr->prov_name = NULL;

    version = FI_VERSION(1, 0);

    err = fi_getinfo(version, NULL, NULL, 0ULL, hints, &fi);
    CHK_OFI_ERR("fi_getinfo", (err < 0), err);

    fi_freeinfo(hints);

    LOG("using OFI provider: %s", fi->fabric_attr->prov_name);

    err = fi_fabric(fi->fabric_attr, &fabric, NULL);
    CHK_OFI_ERR("fi_fabric", (err < 0), err);

    err = fi_domain(fabric, fi, &domain, NULL);
    CHK_OFI_ERR("fi_domain", (err < 0), err);

    av_attr.type = FI_AV_UNSPEC;

    err = fi_av_open(domain, &av_attr, &av, NULL);
    CHK_OFI_ERR("fi_av_open", (err < 0), err);

    for (int i = 0; i < opt.channel_count; i++) {
        cq_attr.format = FI_CQ_FORMAT_TAGGED;
        cq_attr.size = 100;

        err = fi_cq_open(domain, &cq_attr, &ch[i].cq, NULL);
        CHK_OFI_ERR("fi_cq_open", (err < 0), err);

        err = fi_endpoint(domain, fi, &ch[i].ep, NULL);
        CHK_OFI_ERR("fi_endpoint", (err < 0), err);

        err = fi_ep_bind(ch[i].ep, (fid_t)ch[i].cq, FI_SEND | FI_RECV);
        CHK_OFI_ERR("fi_ep_bind cq", (err < 0), err);

        err = fi_ep_bind(ch[i].ep, (fid_t)av, 0);
        CHK_OFI_ERR("fi_ep_bind av", (err < 0), err);

        err = fi_enable(ch[i].ep);
        CHK_OFI_ERR("fi_enable", (err < 0), err);

        memset(local_ep_name, 0, sizeof(local_ep_name));
        memset(remote_ep_name, 0, sizeof(remote_ep_name));

        size_t name_len = EP_NAME_MAX_LEN;
        err = fi_getname((fid_t)ch[i].ep, local_ep_name, &name_len);
        CHK_OFI_ERR("fi_getname", (err < 0), err);

        ASSERT(name_len < EP_NAME_MAX_LEN, "large name_len %zu", name_len);

        if (opt.is_client) {
            send_oob(local_ep_name, EP_NAME_MAX_LEN);
            recv_oob(remote_ep_name, EP_NAME_MAX_LEN);
        }
        else {
            recv_oob(remote_ep_name, EP_NAME_MAX_LEN);
            send_oob(local_ep_name, EP_NAME_MAX_LEN);
        }

        sync_oob();

        struct fi_context av_context;
        size_t ret = fi_av_insert(av,
                                  remote_ep_name,
                                  1, /* count to insert */
                                  &ch[i].peer_addr,
                                  0 /* flags */,
                                  &av_context);
        ASSERT(ret == 1, "av_insert ret should be 1, but got %zu", ret);

        sync_oob();

        LOG("channel #%d is created", i);
    }

    sync_oob();
}

static void finalize_ofi(void) {
    int i;

    for (i = 0; i < opt.channel_count; i++) {
        fi_close((fid_t)ch[i].ep);
        fi_close((fid_t)ch[i].cq);
    }

    fi_close((fid_t)av);
    fi_close((fid_t)domain);
    fi_close((fid_t)fabric);
    fi_freeinfo(fi);
}

static void send_one(int size) {
    int i;

    for (i = 0; i < opt.channel_count; i++)
        SEND_MSG(ch[i].ep, ch[i].sbuf, size, ch[i].peer_addr, &ch[i].sctxt);

    for (i = 0; i < opt.channel_count; i++)
        WAIT_CQ(ch[i].cq, 1);
}

static void recv_one(int size) {
    int i;

    for (i = 0; i < opt.channel_count; i++)
        RECV_MSG(ch[i].ep, ch[i].rbuf, size, ch[i].peer_addr, &ch[i].rctxt);

    for (i = 0; i < opt.channel_count; i++)
        WAIT_CQ(ch[i].cq, 1);
}

static void run_test() {
    int size;
    int i, n, repeat;
    double t1, t2, t;

    for (size = MIN_MSG_SIZE; size <= MAX_MSG_SIZE; size = size << 1) {
        repeat = 1000;
        n = size >> 16;
        while (n) {
            repeat >>= 1;
            n >>= 1;
        }

        printf("send/recv %-8d (x %4d): ", size, repeat);
        fflush(stdout);

        fill_buffers(size);

        t1 = when();
        for (i = 0; i < repeat; i++) {
            if (opt.is_client) {
                recv_one(size);
                send_one(size);
            }
            else {
                send_one(size);
                recv_one(size);
            }
        }
        t2 = when();
        t = (t2 - t1) / repeat / 2;
        printf("%8.2lf us, %8.2lf MB/s, total %8.2lf MS/s\n",
               t,
               size / t,
               size * opt.channel_count / t);

        check_buffers(size);
    }
}

void print_usage(const char* app) {
    printf("usage: %s [-c <channel_count>][-a <alloc_type>][server_name][server_port]\n", app);
    printf("options:\n");
    printf("\t-c <channel_count> number of parallel comm channels [1]\n");
    printf("\t-a <alloc_type> allocation type for comm buffers, runtime|regular [regular]\n");
}

int main(int argc, char* argv[]) {
    int c;

    while ((c = getopt(argc, argv, "c:a:")) != -1) {
        switch (c) {
            case 'c':
                opt.channel_count = atoi(optarg);
                if (opt.channel_count <= 0 || opt.channel_count > MAX_NUM_CHANNELS) {
                    printf("the number of channels must be 1~%d\n", MAX_NUM_CHANNELS);
                    exit(RET_FAILURE);
                }
                break;
            case 'a':
                if (strcmp(optarg, "runtime") == 0) {
                    opt.alloc_type = ALLOC_RUNTIME;
                }
                else if (strcmp(optarg, "regular") == 0) {
                    opt.alloc_type = ALLOC_REGULAR;
                }
                else {
                    print_usage(argv[0]);
                    exit(RET_FAILURE);
                }
                break;

            default:
                print_usage(argv[0]);
                exit(RET_FAILURE);
                break;
        }
    }

    if (argc > optind) {
        opt.is_client = 1;
        opt.server_name = strdup(argv[optind]);
        opt.server_port = atoi(argv[optind + 1]);
    }

    print_options();

    LOG("init_oob");
    init_oob();

    LOG("init_ofi");
    init_ofi();

    if (opt.alloc_type == ALLOC_RUNTIME) {
        LOG("create_gpu_queue");
        gpu_queue = create_gpu_queue();
    }

    LOG("alloc_buffers");
    alloc_buffers();

    LOG("run_test");
    run_test();

    LOG("free_buffers");
    free_buffers();

    LOG("finalize_ofi");
    finalize_ofi();

    LOG("finalize_oob");
    finalize_oob();

    return RET_SUCCESS;
}
