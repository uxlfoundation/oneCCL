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

#ifndef KVS_KEEPER_H_INCLUDED
#define KVS_KEEPER_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif
typedef enum storage_type {
    ST_CLIENT = 0,
    ST_SERVER = 1,
} storage_type_t;

size_t get_count(const char kvs_name[], storage_type_t st_type);

size_t get_val(const char kvs_name[], const char kvs_key[], char* kvs_val, storage_type_t st_type);

size_t get_keys_values(const char* kvs_name,
                       char*** kvs_keys,
                       char*** kvs_values,
                       storage_type_t st_type);

size_t remove_val(const char kvs_name[], const char kvs_key[], storage_type_t st_type);

void put_key(const char kvs_name[],
             const char kvs_key[],
             const char kvs_val[],
             storage_type_t st_type);

void kvs_keeper_clear(storage_type_t st_type);

size_t cut_head(char* kvs_name, char* kvs_key, char* kvs_val, storage_type_t st_type);

size_t get_kvs_list_size(storage_type_t st_type);

#ifdef __cplusplus
}
#endif
#endif
