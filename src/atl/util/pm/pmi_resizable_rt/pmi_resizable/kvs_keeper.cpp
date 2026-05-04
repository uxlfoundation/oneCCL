/*
 Copyright 2016-2020 Intel Corporation
 
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
#include <stdlib.h>
#include <string.h>

#include "util/pm/pmi_resizable_rt/pmi_resizable/helper.hpp"
#include "kvs_keeper.hpp"
#include "def.h"

#define COMPARE_STR(str1, str2, str2_len) (strstr((str1), (str2)) && (strlen(str1) == (str2_len)))

typedef struct kvs_store {
    char name[MAX_KVS_NAME_LENGTH];
    char key[MAX_KVS_KEY_LENGTH];
    char val[MAX_KVS_VAL_LENGTH];
} kvs_store_t;

typedef struct kvs_store_list {
    kvs_store_t kvs;
    struct kvs_store_list* next;
} kvs_store_list_t;

static kvs_store_list_t* head[] = { NULL, NULL };
static size_t kvs_list_size[] = { 0, 0 };

size_t get_count(const char kvs_name[], storage_type_t st_type) {
    size_t count_names = 0;
    size_t i;
    kvs_store_list_t* new_key_ptr = head[st_type];
    size_t kvs_name_len = strlen(kvs_name);
    for (i = 0; i < kvs_list_size[st_type]; i++) {
        if (COMPARE_STR(new_key_ptr->kvs.name, kvs_name, kvs_name_len)) {
            count_names++;
        }
        new_key_ptr = new_key_ptr->next;
    }
    return count_names;
}

size_t get_val(const char kvs_name[], const char kvs_key[], char* kvs_val, storage_type_t st_type) {
    size_t i;
    kvs_store_list_t* new_key_ptr = head[st_type];
    size_t kvs_name_len = strlen(kvs_name);
    size_t kvs_key_len = strlen(kvs_key);
    for (i = 0; i < kvs_list_size[st_type]; i++) {
        if (COMPARE_STR(new_key_ptr->kvs.name, kvs_name, kvs_name_len) &&
            COMPARE_STR(new_key_ptr->kvs.key, kvs_key, kvs_key_len)) {
            kvs_str_copy(kvs_val, new_key_ptr->kvs.val, MAX_KVS_VAL_LENGTH);
            return 1;
        }
        new_key_ptr = new_key_ptr->next;
    }
    return 0;
}

size_t get_keys_values(const char* kvs_name,
                       char*** kvs_keys,
                       char*** kvs_values,
                       storage_type_t st_type) {
    size_t count = 0;
    size_t i;
    kvs_store_list_t* new_key_ptr = head[st_type];
    size_t kvs_name_len = strlen(kvs_name);
    for (i = 0; i < kvs_list_size[st_type]; i++) {
        if (COMPARE_STR(new_key_ptr->kvs.name, kvs_name, kvs_name_len)) {
            count++;
        }
        new_key_ptr = new_key_ptr->next;
    }

    if (count == 0)
        return count;

    if (*kvs_keys != NULL) {
        free(*kvs_keys);
    }

    if (*kvs_values != NULL) {
        free(*kvs_values);
    }

    *kvs_values = (char**)malloc(sizeof(char*) * count);
    if (*kvs_values == NULL) {
        throw std::runtime_error("Memory allocation failed at for kvs_values: " __FILE__ ":" +
                                 std::to_string(__LINE__));
    }
    *kvs_keys = (char**)malloc(sizeof(char*) * count);
    if (*kvs_keys == NULL) {
        throw std::runtime_error("Memory allocation failed at for kvs_keys: " __FILE__ ":" +
                                 std::to_string(__LINE__));
    }

    for (i = 0; i < count; i++) {
        (*kvs_keys)[i] = (char*)malloc(sizeof(char) * MAX_KVS_KEY_LENGTH);
        (*kvs_values)[i] = (char*)malloc(sizeof(char) * MAX_KVS_VAL_LENGTH);
    }

    new_key_ptr = head[st_type];
    for (i = 0; ((new_key_ptr != NULL) && (i < count));) {
        if (COMPARE_STR(new_key_ptr->kvs.name, kvs_name, kvs_name_len)) {
            kvs_str_copy((*kvs_keys)[i], new_key_ptr->kvs.key, MAX_KVS_KEY_LENGTH);
            kvs_str_copy((*kvs_values)[i], new_key_ptr->kvs.val, MAX_KVS_VAL_LENGTH);
            i++;
        }
        new_key_ptr = new_key_ptr->next;
    }
    return count;
}

size_t remove_val(const char kvs_name[], const char kvs_key[], storage_type_t st_type) {
    size_t i;
    kvs_store_list_t* cur_key_ptr = head[st_type];
    kvs_store_list_t* prev_key_ptr = cur_key_ptr;
    size_t kvs_name_len = strlen(kvs_name);
    size_t kvs_key_len = strlen(kvs_key);
    for (i = 0; i < kvs_list_size[st_type]; i++) {
        if (COMPARE_STR(cur_key_ptr->kvs.name, kvs_name, kvs_name_len) &&
            COMPARE_STR(cur_key_ptr->kvs.key, kvs_key, kvs_key_len)) {
            if (cur_key_ptr == head[st_type]) {
                head[st_type] = head[st_type]->next;
            }
            else {
                prev_key_ptr->next = cur_key_ptr->next;
            }
            free(cur_key_ptr);
            kvs_list_size[st_type]--;
            return 0;
        }
        prev_key_ptr = cur_key_ptr;
        cur_key_ptr = cur_key_ptr->next;
    }
    return 1;
}

void put_key(const char kvs_name[],
             const char kvs_key[],
             const char kvs_val[],
             storage_type_t st_type) {
    kvs_store_list_t* tmp_key_ptr = head[st_type];

    if (tmp_key_ptr == NULL) {
        head[st_type] = (kvs_store_list_t*)malloc(sizeof(kvs_store_list_t));
        head[st_type]->next = NULL;
        tmp_key_ptr = head[st_type];
    }
    else {
        kvs_store_list_t* prev_key_ptr = tmp_key_ptr;
        size_t kvs_name_len = strlen(kvs_name);
        size_t kvs_key_len = strlen(kvs_key);
        while (tmp_key_ptr != NULL) {
            if (COMPARE_STR(tmp_key_ptr->kvs.name, kvs_name, kvs_name_len) &&
                COMPARE_STR(tmp_key_ptr->kvs.key, kvs_key, kvs_key_len)) {
                goto copy;
            }

            prev_key_ptr = tmp_key_ptr;
            tmp_key_ptr = tmp_key_ptr->next;
        }
        tmp_key_ptr = prev_key_ptr;

        tmp_key_ptr->next = (kvs_store_list_t*)malloc(sizeof(kvs_store_list_t));
        tmp_key_ptr = tmp_key_ptr->next;
        tmp_key_ptr->next = NULL;
    }
    kvs_list_size[st_type]++;
copy:
    kvs_str_copy(tmp_key_ptr->kvs.name, kvs_name, MAX_KVS_NAME_LENGTH);
    kvs_str_copy(tmp_key_ptr->kvs.key, kvs_key, MAX_KVS_KEY_LENGTH);
    kvs_str_copy(tmp_key_ptr->kvs.val, kvs_val, MAX_KVS_VAL_LENGTH);

    if (strlen(kvs_name) > MAX_KVS_NAME_LENGTH) {
        tmp_key_ptr->kvs.name[MAX_KVS_NAME_LENGTH - 1] = NULL_CHAR;
    }
    if (strlen(kvs_key) > MAX_KVS_KEY_LENGTH) {
        tmp_key_ptr->kvs.key[MAX_KVS_KEY_LENGTH - 1] = NULL_CHAR;
    }
    if (strlen(kvs_val) > MAX_KVS_VAL_LENGTH) {
        tmp_key_ptr->kvs.val[MAX_KVS_VAL_LENGTH - 1] = NULL_CHAR;
    }
}

void kvs_keeper_clear(storage_type_t st_type) {
    kvs_store_list_t* key_ptr;

    while (head[st_type] != NULL) {
        key_ptr = head[st_type];
        head[st_type] = head[st_type]->next;

        free(key_ptr);
        kvs_list_size[st_type]--;
    }
}

size_t cut_head(char* kvs_name, char* kvs_key, char* kvs_val, storage_type_t st_type) {
    kvs_store_list_t* key_ptr = head[st_type];

    if (head[st_type] != NULL) {
        head[st_type] = head[st_type]->next;

        memset(kvs_name, 0, MAX_KVS_NAME_LENGTH);
        memset(kvs_key, 0, MAX_KVS_KEY_LENGTH);
        memset(kvs_val, 0, MAX_KVS_VAL_LENGTH);
        kvs_str_copy(kvs_name, key_ptr->kvs.name, MAX_KVS_NAME_LENGTH);
        kvs_str_copy(kvs_key, key_ptr->kvs.key, MAX_KVS_KEY_LENGTH);
        kvs_str_copy(kvs_val, key_ptr->kvs.val, MAX_KVS_VAL_LENGTH);

        free(key_ptr);
        kvs_list_size[st_type]--;
        return 1;
    }
    return 0;
}

size_t get_kvs_list_size(storage_type_t st_type) {
    return kvs_list_size[st_type];
}
