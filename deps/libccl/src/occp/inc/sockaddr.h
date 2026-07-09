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

#include <netinet/in.h> // for sockaddr_in, sockaddr_in6
#include <arpa/inet.h> // for inet_ntoa, inet_ntop, inet_pton
#include <string>
#include <vector>
#include <cstring>

// automatic IPv4/v6 handling of sockaddr_*

class sockaddr_str_t {
public:
    sockaddr_str_t(const sockaddr_storage& address) {
        set(address);
    }
    sockaddr_str_t& operator=(const sockaddr_storage& address) {
        return set(address);
    }

    operator const std::string&() const {
        return m_str;
    }

private:
    sockaddr_str_t& set(const sockaddr_storage& address);

    std::string m_str;
};

class sockaddr_t {
public:
    sockaddr_t();
    sockaddr_t(const sockaddr_t& addr);
    sockaddr_t(const sockaddr_storage& addr);
    sockaddr_t(const char* ipaddress, in_port_t _port = 0);

    sockaddr_t& operator=(const sockaddr_t& addr);
    sockaddr_t& operator=(const sockaddr_storage& addr);
    sockaddr_t& operator=(const char* ipaddress);

    operator const sockaddr*() const noexcept {
        return reinterpret_cast<const sockaddr*>(&m_sockAddr);
    }
    operator sockaddr*() noexcept {
        return reinterpret_cast<sockaddr*>(&m_sockAddr);
    }
    operator const sockaddr_storage&() const noexcept {
        return m_sockAddr;
    }
    operator sa_family_t() const noexcept {
        return m_sockAddr.ss_family;
    }
    operator socklen_t() const noexcept {
        return size_of();
    }

    std::string addr() const; // returns IP address as string (no port)
    std::string str() const; // returns IP address and port as string addr:port
    operator std::string() const {
        return str();
    }
    in_port_t port() const noexcept;
    void port(in_port_t _port) noexcept;
    socklen_t size_of() const noexcept;

    void reset() {
        m_sockAddr = {};
        m_sockAddr.ss_family = AF_INET; // default to IPv4
    }

private:
    sockaddr_storage m_sockAddr = {};

    bool IPv4() const noexcept {
        return m_sockAddr.ss_family == AF_INET;
    }
    sockaddr_in* sa4() noexcept {
        return reinterpret_cast<sockaddr_in*>(&m_sockAddr);
    }
    const sockaddr_in* sa4() const noexcept {
        return reinterpret_cast<const sockaddr_in*>(&m_sockAddr);
    }
    sockaddr_in6* sa6() noexcept {
        return reinterpret_cast<sockaddr_in6*>(&m_sockAddr);
    }
    const sockaddr_in6* sa6() const noexcept {
        return reinterpret_cast<const sockaddr_in6*>(&m_sockAddr);
    }

    void fromString(const char* ipaddress);
};

using caddrs_t = std::vector<sockaddr_storage>;

static inline bool operator==(const struct in6_addr& a, const struct in6_addr& b) {
    return std::memcmp(&a, &b, sizeof(in6_addr)) == 0;
}

static inline bool operator==(const sockaddr_storage& a, const sockaddr_storage& b) {
    return std::memcmp(&a, &b, sizeof(sockaddr_storage)) == 0;
}

// compare address only (no port)
static inline bool same_address(const sockaddr_storage& a, const sockaddr_storage& b) {
    if (a.ss_family == b.ss_family) {
        if (a.ss_family == AF_INET) {
            return reinterpret_cast<const sockaddr_in&>(a).sin_addr.s_addr ==
                   reinterpret_cast<const sockaddr_in&>(b).sin_addr.s_addr;
        }
        else if (a.ss_family == AF_INET6) {
            return reinterpret_cast<const sockaddr_in6&>(a).sin6_addr ==
                   reinterpret_cast<const sockaddr_in6&>(b).sin6_addr;
        }
    }
    return false;
}
