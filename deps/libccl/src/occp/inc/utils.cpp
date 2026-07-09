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

#include "utils.h"
#include "sockaddr.h"
#include <ifaddrs.h>
#include <vector>
#include <string.h>

std::vector<std::string> get_bound_interfaces() {
    std::vector<std::string> interfaces;

    struct ifaddrs* ifap;
    if (getifaddrs(&ifap) == 0) {
        for (struct ifaddrs* ifa = ifap; ifa != nullptr; ifa = ifa->ifa_next) {
            if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET) {
                struct sockaddr_in* sin = (struct sockaddr_in*)ifa->ifa_addr;
                char ip_str[INET_ADDRSTRLEN];

                if (inet_ntop(AF_INET, &sin->sin_addr, ip_str, INET_ADDRSTRLEN)) {
                    // Skip loopback interface unless specifically needed
                    if (strcmp(ip_str, "127.0.0.1") != 0) {
                        std::string iface_info = std::string(ifa->ifa_name) + " (" + ip_str + ")";
                        interfaces.push_back(iface_info);
                    }
                }
            }
        }
        freeifaddrs(ifap);
    }

    return interfaces;
}
