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

#include "sockaddr.h"
#include "utils.h"

sockaddr_str_t& sockaddr_str_t::set(const sockaddr_storage& address) {
    char str_addr[INET6_ADDRSTRLEN] = {};
    const char* ptr = nullptr;
    in_port_t port = 0;

    if (AF_INET == address.ss_family) {
        const auto* addr = reinterpret_cast<const sockaddr_in*>(&address);
        ptr = inet_ntop(AF_INET, &addr->sin_addr, str_addr, sizeof(str_addr));
        port = ntohs(addr->sin_port);
    }
    else if (AF_INET6 == address.ss_family) {
        const auto* addr = reinterpret_cast<const sockaddr_in6*>(&address);
        ptr = inet_ntop(AF_INET6, &addr->sin6_addr, str_addr, sizeof(str_addr));
        port = ntohs(addr->sin6_port);
    }

    if (ptr) {
        m_str.clear();
        m_str.reserve(INET6_ADDRSTRLEN + 8); // Reserve space for IP + ":" + port
        m_str.assign(ptr);
        m_str.push_back(':');
        m_str.append(std::to_string(port));
    }
    else {
        m_str = "invalid ip address";
    }

    return *this;
}

sockaddr_t::sockaddr_t() {
    m_sockAddr.ss_family = AF_INET; // default to IPv4
}

sockaddr_t::sockaddr_t(const sockaddr_t& addr) {
    m_sockAddr = addr.m_sockAddr;
}

sockaddr_t& sockaddr_t::operator=(const sockaddr_t& addr) {
    m_sockAddr = addr.m_sockAddr;
    return *this;
}

sockaddr_t::sockaddr_t(const sockaddr_storage& addr) {
    m_sockAddr = addr;
}

sockaddr_t& sockaddr_t::operator=(const sockaddr_storage& addr) {
    m_sockAddr = addr;
    return *this;
}

sockaddr_t::sockaddr_t(const char* ipaddress, in_port_t _port) {
    fromString(ipaddress);
    port(_port);
}

sockaddr_t& sockaddr_t::operator=(const char* ipaddress) {
    fromString(ipaddress);
    return *this;
}

void sockaddr_t::fromString(const char* ipaddress) {
    if (!ipaddress || ipaddress[0] == '\0') {
        m_sockAddr = {};
        m_sockAddr.ss_family = AF_INET;
        return;
    }

    if (inet_pton(AF_INET, ipaddress, &(sa4()->sin_addr)) == 1) {
        // IPv4 address
        m_sockAddr.ss_family = AF_INET;
    }
    else if (inet_pton(AF_INET6, ipaddress, &(sa6()->sin6_addr)) == 1) {
        // IPv6 address
        m_sockAddr.ss_family = AF_INET6;
    }
    else {
        VERIFY(false, "invalid ip string: {}", ipaddress);
    }
}

socklen_t sockaddr_t::size_of() const noexcept {
    return IPv4() ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
}

in_port_t sockaddr_t::port() const noexcept {
    return IPv4() ? ntohs(sa4()->sin_port) : ntohs(sa6()->sin6_port);
}

void sockaddr_t::port(in_port_t _port) noexcept {
    IPv4() ? sa4()->sin_port = htons(_port) : sa6()->sin6_port = htons(_port);
}

std::string sockaddr_t::str() const {
    return sockaddr_str_t(m_sockAddr);
}

std::string sockaddr_t::addr() const {
    char str_addr[INET6_ADDRSTRLEN] = {};
    const char* ptr = nullptr;

    if (AF_INET == m_sockAddr.ss_family) {
        ptr = inet_ntop(AF_INET, &(sa4()->sin_addr), str_addr, sizeof(str_addr));
    }
    else if (AF_INET6 == m_sockAddr.ss_family) {
        ptr = inet_ntop(AF_INET6, &(sa6()->sin6_addr), str_addr, sizeof(str_addr));
    }

    return ptr ? std::string(ptr) : "invalid";
}
