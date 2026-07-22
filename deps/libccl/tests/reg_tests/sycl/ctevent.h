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
#include <fstream>
#include <mutex>
#include <atomic>
#include <stack>
#include <string>
#include <unistd.h>
#include <vector>
#include <unordered_set>
#include <thread>

namespace CTEvent {
inline bool &enabled() {
    static bool _enabled =
        ((getenv("CTENABLE") != nullptr) && (std::string(getenv("CTENABLE")) == "1"));
    return _enabled;
}
struct Config {
    std::string tid = "tid";
    std::string pid = std::to_string(getpid());
    std::string logname = "ctevents.json";
    std::string args = "";
    bool operator==(const Config &rhs) {
        return logname == rhs.logname && tid == rhs.tid;
    }
};
inline Config config(Config config = {}) {
    static Config _config;
    if (!enabled())
        return _config;
    if (config == Config{})
        return _config;
    _config = config;
    return _config;
}
inline void setconfig(std::string uniquename) {
    if (!enabled())
        return;
    config({ uniquename, std::to_string(getpid()), uniquename + ".json", "" });
}
struct Event {
    std::string name;
    std::chrono::system_clock::time_point start;
    std::chrono::duration<double> dur;
    std::string tid;
    std::string pid;
    std::string args;
};

inline void flush();

struct CTEvent {
    CTEvent() {}
    static CTEvent &instance() {
        static CTEvent _queue{};
        return _queue;
    }
    ~CTEvent() {
        flush();
    }

    std::vector<Event> queue;
    std::mutex lock;
};
inline std::stack<Event> &tstack() {
    thread_local std::stack<Event> _tstack;
    return _tstack;
}
inline std::string &tid() {
    thread_local std::string _tid = "";
    if (_tid == "") {
        auto id = std::this_thread::get_id();
        std::stringstream ss;
        ss << id;
        _tid = ss.str();
    }
    return _tid;
}

inline void push(std::string name, std::string args, std::string tid, std::string pid) {
    if (!enabled())
        return;
    std::unique_lock<std::mutex> _(CTEvent::instance().lock);
    Event e{
        name, std::chrono::high_resolution_clock::now(), std::chrono::duration<double>(0), tid, pid,
        args
    };
    tstack().push(std::move(e));
};
inline void push(std::string name) {
    auto c = config();
    push(name, c.args, c.tid + "_" + tid(), c.pid);
}
inline void push(std::string name, std::string args) {
    auto c = config();
    push(name, args, c.tid + "_" + tid(), c.pid);
}
inline void push(std::string name, std::string args, std::string tid) {
    auto c = config();
    push(name, args, tid, c.pid);
}
inline void pop() {
    if (!enabled())
        return;
    if (tstack().empty())
        return;
    Event e = tstack().top();
    tstack().pop();
    e.dur = std::chrono::high_resolution_clock::now() - e.start;

    std::unique_lock<std::mutex> _(CTEvent::instance().lock);
    CTEvent::instance().queue.push_back(std::move(e));
}
inline void flush() {
    if (!enabled())
        return;
    std::unique_lock<std::mutex> _(CTEvent::instance().lock);
    if (CTEvent::instance().queue.empty())
        return;
    std::ofstream out(config().logname);
    out << "[\n";
    bool begin = true;
    for (auto &e : CTEvent::instance().queue) {
        std::string str = "";
        if (!begin)
            str += ",";
        else
            begin = false;
        str += R"({"name":")" + e.name + R"(","ph":"X","pid":")" + e.pid + R"(","tid":")" + e.tid +
               R"(","ts":)" +
               std::to_string(
                   std::chrono::duration_cast<std::chrono::microseconds>(e.start.time_since_epoch())
                       .count()) +
               R"(,"dur":)" +
               std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(e.dur).count());
        if (!e.args.empty()) {
            str += R"(,"args":)" + e.args;
        }
        str += "}\n";
        out << str;
    }
    out << "\n]\n";
    CTEvent::instance().queue.clear();
}
struct Scoped {
    Scoped(std::string name, std::string tid, std::string pid, std::string args) {
        push(name, tid, pid, args);
    }
    Scoped(std::string name) {
        push(name);
    }
    Scoped(std::string name, std::string args) {
        push(name, args);
    }
    ~Scoped() {
        pop();
    }
};

inline std::vector<double> get_stats(std::string name) {
    std::vector<Event> vec;
    std::copy_if(CTEvent::instance().queue.begin(),
                 CTEvent::instance().queue.end(),
                 std::back_inserter(vec),
                 [&](const Event &a) {
                     return a.name == name;
                 });
    if (vec.size() == 0)
        return {};

    auto max = *max_element(vec.begin(), vec.end(), [](const Event &a, const Event &b) {
        return a.dur < b.dur;
    });
    auto min = *max_element(vec.begin(), vec.end(), [](const Event &a, const Event &b) {
        return a.dur > b.dur;
    });
    double avg =
        std::accumulate(
            vec.begin(),
            vec.end(),
            0.0,
            [](int sum, const Event &e) {
                return sum + std::chrono::duration_cast<std::chrono::microseconds>(e.dur).count();
            }) /
        vec.size();

    double accum = 0.0;
    std::for_each(vec.begin(), vec.end(), [&](const Event &e) {
        accum += (std::chrono::duration_cast<std::chrono::microseconds>(e.dur).count() - avg) *
                 (std::chrono::duration_cast<std::chrono::microseconds>(e.dur).count() - avg);
    });
    double stddev = std::sqrt(accum / (vec.size() - 1));

    return { (double)std::chrono::duration_cast<std::chrono::microseconds>(min.dur).count(),
             double(std::chrono::duration_cast<std::chrono::microseconds>(max.dur).count()),
             avg,
             stddev };
}

inline void print_stats(std::string total = "iteration") {
    std::unordered_set<std::string> names;
    for (auto &e : CTEvent::instance().queue) {
        names.insert(e.name);
    }
    std::string last = "";
    if (names.count(total) > 0)
        last = total;
    for (auto &n : names) {
        if (n != last) {
            auto r = get_stats(n);
            std::cout << n << ": " << r[0] << ", " << r[1] << ", " << r[2] << std::endl;
        }
    }
    if (last != "") {
        auto r = get_stats(last);
        std::cout << last << ": " << r[0] << ", " << r[1] << ", " << r[2] << std::endl;
    }
}
inline int forkcmd() {
    const int limit = 100;
    auto cmd = getenv("CTCMD");
    if (cmd == nullptr)
        return 0;
    char *args[limit] = {};
    args[0] = strtok(cmd, " ");
    int i = 0;
    while (args[i++] != NULL) {
        if (i == limit)
            return 0;
        args[i] = strtok(NULL, " ");
    }
    int pid = fork();
    if (pid == 0) {
        if ((execvp(args[0], args)) < 0) {
            std::cout << "error running ctcmd" << std::endl;
        }
        exit(0);
    }
    else
        return pid;
}
} // namespace CTEvent
