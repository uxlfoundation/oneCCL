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

#include <chrono>
#include <mutex>
#include <string>
#include <sys/file.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define CHECK(ret, msg) \
    if ((ret) < 0) { \
        throw std::system_error(errno, std::system_category(), msg); \
    }

class base_store {
public:
    base_store(){};

    virtual ~base_store(){};

    virtual int write(const void* data, size_t size) = 0;

    virtual int read(void* data, size_t size) = 0;
};

class file_store : public base_store {
public:
    file_store(const file_store& other) = delete;
    file_store& operator=(const file_store& other) = delete;
    file_store(std::string path, int rank, const std::chrono::seconds& timeout)
            : base_store(),
              path(path),
              rank(rank),
              pos(0),
              fd(-1),
              timeout(timeout){};

    virtual ~file_store() {
        if (rank == 0)
            std::remove(path.c_str());
    };

    void release_resources() {
        try {
            CHECK(flock(fd, LOCK_UN), "Unlocking file: ");
        }
        catch (const std::system_error& e) {
            fprintf(stderr, "%d\n%s\n", e.code().value(), e.what());
        }

        close(fd);
        fd = -1;
    }

    int write(const void* data, size_t size) override {
        int ret = 0;
        std::unique_lock<std::mutex> locker(mtx);
        fd = open(path.c_str(), O_CREAT | O_RDWR, 0644);
        CHECK(fd, "Open file to write into (" + path + "): ");

        try {
            CHECK(flock(fd, LOCK_EX), "Setting exclusive rights for writing to the file: ");
            CHECK(lseek(fd, 0, SEEK_END), "Setting a cursor at the EOF: ");

            // writing into the file
            while (size > 0) {
                auto wr_v = ::write(fd, data, size);
                CHECK(wr_v, "An error occured while writing to the file: ");
                data = (uint8_t*)data + wr_v;
                size -= wr_v;
            }
            CHECK(fsync(fd), "Flushing file content: ");
        }
        catch (const std::system_error& e) {
            fprintf(stderr, "%d\n%s\n", e.code().value(), e.what());
            ret = -1;
        }

        release_resources();
        return ret;
    };

    int read(void* data, size_t size) override {
        const auto time_start = std::chrono::steady_clock::now();
        while (1) {
            std::unique_lock<std::mutex> locker(mtx);
            fd = open(path.c_str(), O_RDONLY);
            if (fd < 0 && errno == ENOENT) {
                // file might not exist yet
                const auto time_passed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - time_start);
                if (time_passed > timeout) {
                    throw std::runtime_error("Timeout " + std::to_string(timeout.count()) +
                                             "s waiting for the file " + path + " to open");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10 * rank));
                continue;
            }
            else {
                CHECK(fd, "Open the file to read from (" + path + "): ");
            }

            try {
                CHECK(flock(fd, LOCK_SH), "Setting shared rights for reading the file: ");

                auto start = lseek(fd, 0, SEEK_SET);
                CHECK(start, "Setting the cursor at the beginning of the file: ");

                // find the real size of the file
                auto len = lseek(fd, 0, SEEK_END);
                CHECK(len, "Setting the cursor at the EOF: ");

                if (len == start) {
                    // nothing has been written yet
                    release_resources();
                    locker.unlock();
                    const auto time_passed = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now() - time_start);
                    if (time_passed > timeout) {
                        throw std::runtime_error("Timeout " + std::to_string(timeout.count()) +
                                                 "s waiting for the file " + path + " to read");
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(10 * rank));
                    continue;
                }

                // start from where we stopped last time
                start = lseek(fd, pos, SEEK_SET);
                CHECK(start, "Setting the cursor at the last known position: ");

                // if there are still some bytes to read
                if (len > start && size > 0) {
                    size -= len;
                    while (len > 0) {
                        auto rd = ::read(fd, data, len);
                        CHECK(rd, "An error occured while reading the file: ")
                        data = (uint8_t*)data + rd;
                        len -= rd;
                    }
                    pos = lseek(fd, 0, SEEK_CUR);
                    CHECK(pos, "Saving the cursor current position: ");
                }
                else {
                    release_resources();
                    break;
                }
            }
            catch (const std::system_error& e) {
                fprintf(stderr, "%d\n%s\n", e.code().value(), e.what());
                release_resources();
                return -1;
            }
        }
        return 0;
    };

protected:
    std::string path;
    int rank;
    off_t pos;
    int fd;
    std::chrono::seconds timeout;
    std::mutex mtx;
};
