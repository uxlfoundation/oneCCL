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

#pragma once

#include <cstdint>

#include "util/pm/pmi_resizable_rt/pmi_resizable/def.h"
#include "internal_kvs.h"

/// kvs_access_mode_t is serialized to 4-byte-long buffer, as little endian
/// uint32_t is chosen:
/// 1. There is a bijective relation uint32_t -> uint8_t[4]
/// 2. No corner case handling for negative values is required
enum kvs_access_mode_t : uint32_t {
    AM_CLOSE = 1,
    AM_PUT = 2,
    AM_REMOVE = 3,
    AM_GET_COUNT = 4,
    AM_GET_VAL = 5,
    AM_GET_KEYS_VALUES = 6,
    AM_GET_REPLICA = 7,
    AM_FINALIZE = 8,
    AM_BARRIER = 9,
    AM_BARRIER_REGISTER = 10,
    AM_INTERNAL_REGISTER = 11,
    AM_SET_SIZE = 12,
};

class kvs_request_t {
public:
    kvs_status_t put(int sock,
                     kvs_access_mode_t put_mode,
                     std::mutex& memory_mutex,
                     const std::string& kvs_name = {},
                     const std::string& kvs_key = {},
                     const std::string& kvs_val = {}) {
        std::vector<char> put_buf(put_buf_size, 0);
        size_t step = 0;
        // convert local endianness to little endian
        uint8_t put_mode_buff[4]{
            (static_cast<uint8_t>((static_cast<uint32_t>(put_mode) >> 0) & 0xFF)),
            (static_cast<uint8_t>((static_cast<uint32_t>(put_mode) >> 8) & 0xFF)),
            (static_cast<uint8_t>((static_cast<uint32_t>(put_mode) >> 16) & 0xFF)),
            (static_cast<uint8_t>((static_cast<uint32_t>(put_mode) >> 24) & 0xFF)),
        };
        static_assert(sizeof(put_mode_buff) == sizeof(mode),
                      "`mode` size is no longer compatible with `put_mode_buff`");
        static_assert(sizeof(put_mode_buff) == sizeof(put_mode),
                      "`put_mode` size is no longer compatible with `put_mode_buff`");
        static_assert(sizeof(put_mode_buff) == sizeof(kvs_access_mode_t),
                      "`kvs_access_mode_t` size is no longer compatible with `put_mode_buff`");
        std::copy(std::begin(put_mode_buff), std::end(put_mode_buff), put_buf.begin());

        if (!kvs_name.empty()) {
            KVS_ERROR_IF_NOT(kvs_name.length() <= MAX_KVS_NAME_LENGTH);
            step += sizeof(mode);
            std::copy(kvs_name.begin(), kvs_name.end(), put_buf.begin() + step);
        }

        if (!kvs_key.empty()) {
            KVS_ERROR_IF_NOT(kvs_key.length() <= MAX_KVS_KEY_LENGTH);
            step += sizeof(name);
            std::copy(kvs_key.begin(), kvs_key.end(), put_buf.begin() + step);
        }

        if (!kvs_val.empty()) {
            KVS_ERROR_IF_NOT(kvs_val.length() <= MAX_KVS_VAL_LENGTH);
            step += sizeof(key);
            std::copy(kvs_val.begin(), kvs_val.end(), put_buf.begin() + step);
        }

        DO_RW_OP(write, sock, put_buf.data(), put_buf.size(), memory_mutex);
        return KVS_STATUS_SUCCESS;
    }
    kvs_status_t put(int sock, std::mutex& memory_mutex, size_t put_buf) {
        const size_t sizeof_put_buf = sizeof(put_buf);
        DO_RW_OP(write, sock, &put_buf, sizeof_put_buf, memory_mutex);
        return KVS_STATUS_SUCCESS;
    }
    kvs_status_t put(int sock, std::mutex& memory_mutex, const std::string& put_buf) {
        KVS_ERROR_IF_NOT(put_buf.size() <= MAX_KVS_VAL_LENGTH);
        std::vector<char> tmp_val(MAX_KVS_VAL_LENGTH, 0);
        std::copy(put_buf.begin(), put_buf.end(), tmp_val.begin());

        DO_RW_OP(write, sock, tmp_val.data(), tmp_val.size(), memory_mutex);
        return KVS_STATUS_SUCCESS;
    }
    kvs_status_t put(int sock,
                     std::mutex& memory_mutex,
                     const std::map<std::string, std::string>& requests) {
        std::vector<char> tmp_val((MAX_KVS_KEY_LENGTH + MAX_KVS_VAL_LENGTH) * requests.size(), 0);
        size_t i = 0;
        for (auto& request : requests) {
            KVS_ERROR_IF_NOT(request.first.size() <= MAX_KVS_KEY_LENGTH);
            std::copy(request.first.begin(),
                      request.first.end(),
                      tmp_val.begin() + i * MAX_KVS_KEY_LENGTH);
            KVS_ERROR_IF_NOT(request.second.size() <= MAX_KVS_VAL_LENGTH);
            std::copy(
                request.second.begin(),
                request.second.end(),
                tmp_val.begin() + MAX_KVS_KEY_LENGTH * requests.size() + i * MAX_KVS_VAL_LENGTH);
            i++;
        }

        DO_RW_OP(write, sock, tmp_val.data(), tmp_val.size(), memory_mutex);
        return KVS_STATUS_SUCCESS;
    }
    kvs_status_t get(int sock, std::mutex& memory_mutex, size_t& get_buf) {
        const size_t sizeof_get_buf = sizeof(get_buf);
        DO_RW_OP(read, sock, &get_buf, sizeof_get_buf, memory_mutex);
        return KVS_STATUS_SUCCESS;
    }
    kvs_status_t get(int sock, std::mutex& memory_mutex, std::string& get_buf) {
        get_buf.clear();
        std::vector<char> output_vec(MAX_KVS_VAL_LENGTH, 0);
        DO_RW_OP(read, sock, output_vec.data(), output_vec.size(), memory_mutex);
        std::copy(output_vec.begin(), output_vec.end(), std::back_inserter(get_buf));
        return KVS_STATUS_SUCCESS;
    }

    kvs_status_t get(int sock,
                     std::mutex& memory_mutex,
                     size_t count,
                     std::vector<std::string>& key_buf,
                     std::vector<std::string>& val_buf) {
        std::vector<char> tmp_val(count * (MAX_KVS_KEY_LENGTH + MAX_KVS_VAL_LENGTH), 0);
        DO_RW_OP(read, sock, tmp_val.data(), tmp_val.size(), memory_mutex);
        auto it = tmp_val.begin();
        // if vector is empty then skip processing
        // user doesn't need this result
        if (!key_buf.empty()) {
            key_buf.resize(count);
            for (size_t i = 0; i < count; i++) {
                key_buf[i].resize(MAX_KVS_KEY_LENGTH, 0);
                std::copy(it, it + MAX_KVS_KEY_LENGTH, key_buf[i].begin());
                it += MAX_KVS_KEY_LENGTH;
            }
        }
        if (!val_buf.empty()) {
            val_buf.resize(count);
            it = tmp_val.begin() + count * MAX_KVS_KEY_LENGTH;
            for (size_t i = 0; i < count; i++) {
                val_buf[i].resize(MAX_KVS_VAL_LENGTH, 0);
                std::copy(it, it + MAX_KVS_VAL_LENGTH, val_buf[i].begin());
                it += MAX_KVS_VAL_LENGTH;
            }
        }

        return KVS_STATUS_SUCCESS;
    }
    kvs_status_t get(int sock, std::mutex& memory_mutex) {
        int ret = 0;
        std::vector<char> get_buf(put_buf_size, 0);
        DO_RW_OP_1(read, sock, get_buf.data(), get_buf.size(), ret);
        if (ret == 0) {
            mode = AM_CLOSE;
            return KVS_STATUS_SUCCESS;
        }
        auto it_get_buf = get_buf.begin();
        uint8_t temp_mode[4]{ 0 };
        static_assert(sizeof(temp_mode) == sizeof(kvs_access_mode_t),
                      "`kvs_access_mode_t` size is no longer compatible with `temp_mode`");
        std::copy(it_get_buf, it_get_buf + sizeof(mode), temp_mode);
        // convert little-endian from serialized buffer to host endianness
        mode = static_cast<kvs_access_mode_t>((temp_mode[0] << 0) | (temp_mode[1] << 8) |
                                              (temp_mode[2] << 16) | (temp_mode[3] << 24));
        it_get_buf += sizeof(mode);
        std::copy(it_get_buf, it_get_buf + sizeof(name), name);
        it_get_buf += sizeof(name);
        std::copy(it_get_buf, it_get_buf + sizeof(key), key);
        it_get_buf += sizeof(key);
        std::copy(it_get_buf, it_get_buf + sizeof(val), val);

        return KVS_STATUS_SUCCESS;
    }

private:
    friend class server;
    kvs_access_mode_t mode{ AM_PUT }; // serialized to little-endian byte order for buffer transfers
    char name[MAX_KVS_NAME_LENGTH]{};
    char key[MAX_KVS_KEY_LENGTH]{};
    char val[MAX_KVS_VAL_LENGTH]{};
    const size_t put_buf_size = sizeof(mode) + sizeof(name) + sizeof(key) + sizeof(val);
};

typedef struct server_args {
    int sock_listener;
    std::shared_ptr<isockaddr> args;
} server_args_t;

void* kvs_server_init(void* args);
