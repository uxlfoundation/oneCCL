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

#ifndef LISTENER_H_INCLUDED
#define LISTENER_H_INCLUDED

#include "helper.hpp"

class pmi_listener {
public:
    kvs_status_t send_notification(int sig, std::shared_ptr<helper> h);

    void set_applied_count(int count);

    kvs_status_t run_listener(std::shared_ptr<helper> h);

private:
    kvs_status_t collect_sock_addr(std::shared_ptr<helper> h);
    kvs_status_t clean_listener(std::shared_ptr<helper> h);
};
#endif
