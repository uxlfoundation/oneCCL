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

// Minimal ordered-JSON parser for the collectives selection policy.
//
// Supports the subset the policy files use:
//   - objects (order-preserving), arrays, strings, integers, doubles,
//     true/false/null
//   - `// line` and `/* block */` comments
//   - \" \\ \/ \b \f \n \r \t and \uXXXX with UTF-16 surrogate pairs
//
// Design notes:
//   - Parser walks a raw `const char*` cursor with an end sentinel — no
//     per-char bounds calls, no substr allocation while tokenizing.
//   - Value owns nested containers through unique_ptr, so it is move-only and
//     traversal never invokes atomic refcount ops.
//   - Numbers are parsed directly with overflow detection; overflow of the
//     integer path falls back to double instead of rejecting the file.
//   - Recursion is bounded by kMaxDepth to keep pathological input from
//     blowing the stack.
//
// Non-goals: full RFC 8259 conformance beyond what policy files need,
// streaming / incremental input.

#pragma once

#include <cerrno>
#include <cmath>       // HUGE_VAL
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <istream>
#include <limits>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <clocale>    // newlocale, freelocale, strtod_l (via _GNU_SOURCE)

namespace ccl::selection::mini_json {

class Value;

using Object = std::vector<std::pair<std::string, Value>>;
using Array = std::vector<Value>;

enum class Type : std::uint8_t { Null, Bool, Int, Double, String, Array, Object };

class parse_error : public std::runtime_error {
public:
    parse_error(const char* msg, std::size_t line, std::size_t col)
        : std::runtime_error(format(msg, line, col)),
          line_(line), col_(col) {}

    [[nodiscard]] std::size_t line() const { return line_; }
    [[nodiscard]] std::size_t col() const  { return col_; }

private:
    static std::string format(const char* msg, std::size_t line, std::size_t col) {
        std::string out = "mini_json: ";
        out += msg;
        out += " at line ";
        out += std::to_string(line);
        out += ", col ";
        out += std::to_string(col);
        return out;
    }
    std::size_t line_;
    std::size_t col_;
};

class Value {
public:
    Value() = default;

    Value(const Value&) = delete;
    Value& operator=(const Value&) = delete;
    Value(Value&&) noexcept = default;
    Value& operator=(Value&&) noexcept = default;

    [[nodiscard]] Type type() const              { return type_; }
    [[nodiscard]] bool is_null() const           { return type_ == Type::Null; }
    [[nodiscard]] bool is_boolean() const        { return type_ == Type::Bool; }
    [[nodiscard]] bool is_number_integer() const { return type_ == Type::Int; }
    [[nodiscard]] bool is_number_float() const   { return type_ == Type::Double; }
    [[nodiscard]] bool is_number() const         { return is_number_integer() || is_number_float(); }
    [[nodiscard]] bool is_string() const         { return type_ == Type::String; }
    [[nodiscard]] bool is_array() const          { return type_ == Type::Array; }
    [[nodiscard]] bool is_object() const         { return type_ == Type::Object; }

    // empty() applies to strings, arrays, and objects.
    [[nodiscard]] bool empty() const {
        switch (type_) {
            case Type::String: return str_.empty();
            case Type::Array:  return !arr_ || arr_->empty();
            case Type::Object: return !obj_ || obj_->empty();
            default:           return true;
        }
    }

    // size() applies to strings, arrays, and objects; 0 otherwise.
    [[nodiscard]] std::size_t size() const {
        switch (type_) {
            case Type::String: return str_.size();
            case Type::Array:  return arr_ ? arr_->size() : 0;
            case Type::Object: return obj_ ? obj_->size() : 0;
            default:           return 0;
        }
    }

    // Only the getters actually needed by callers.
    template <class T> T get() const;

    // O(n) lookup — Object is ordered by design so we don't want a map.
    // Returns nullptr if not present or `*this` is not an object.
    [[nodiscard]] const Value* find(std::string_view key) const {
        if (type_ != Type::Object || !obj_) return nullptr;
        for (const auto& kv : *obj_) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }

    [[nodiscard]] std::string dump() const {
        std::ostringstream os;
        dump_to(os);
        return os.str();
    }

    // --- iteration (objects only) -----------------------------------------
    class const_iterator {
    public:
        using inner = Object::const_iterator;
        explicit const_iterator(inner it) : it_(it) {}
        [[nodiscard]] const std::string& key() const { return it_->first; }
        [[nodiscard]] const Value&       value() const { return it_->second; }
        const Value&    operator*() const  { return it_->second; }
        const Value*    operator->() const { return &it_->second; }
        const_iterator& operator++()       { ++it_; return *this; }
        bool operator==(const const_iterator& o) const { return it_ == o.it_; }
        bool operator!=(const const_iterator& o) const { return it_ != o.it_; }
    private:
        inner it_;
    };

    // One shared static so begin()/end() return iterators into the SAME
    // container when the value is not an object (checked-iterator builds
    // otherwise abort on mixed containers).
    static const Object& empty_object() {
        static const Object kEmpty;
        return kEmpty;
    }
    [[nodiscard]] const_iterator begin() const {
        return const_iterator((obj_ ? *obj_ : empty_object()).begin());
    }
    [[nodiscard]] const_iterator end() const {
        return const_iterator((obj_ ? *obj_ : empty_object()).end());
    }

    static const Array& empty_array() {
        static const Array kEmpty;
        return kEmpty;
    }
    [[nodiscard]] const Object& as_object() const {
        return obj_ ? *obj_ : empty_object();
    }
    [[nodiscard]] const Array& as_array() const {
        return arr_ ? *arr_ : empty_array();
    }

    // --- factories --------------------------------------------------------
    static Value parse(std::istream& in) {
        std::string src((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
        return parse(src);
    }

    static Value parse(std::string_view src) {
        Parser p(src);
        p.skip_ws();
        Value v = p.parse_value(/*depth*/ 0);
        p.skip_ws();
        if (!p.at_end()) p.fail("trailing content after JSON value");
        return v;
    }

private:
    static constexpr int kMaxDepth = 256;

    Type type_ = Type::Null;
    bool bool_ = false;
    long long int_ = 0;
    double dbl_ = 0.0;
    std::string str_;
    std::unique_ptr<Array>  arr_;
    std::unique_ptr<Object> obj_;

    static Value make_bool(bool b)          { Value v; v.type_ = Type::Bool;   v.bool_ = b; return v; }
    static Value make_int(long long i)      { Value v; v.type_ = Type::Int;    v.int_  = i; return v; }
    static Value make_double(double d)      { Value v; v.type_ = Type::Double; v.dbl_  = d; return v; }
    static Value make_string(std::string s) { Value v; v.type_ = Type::String; v.str_  = std::move(s); return v; }
    static Value make_array(Array a)        { Value v; v.type_ = Type::Array;
                                              v.arr_ = std::make_unique<Array>(std::move(a));  return v; }
    static Value make_object(Object o)      { Value v; v.type_ = Type::Object;
                                              v.obj_ = std::make_unique<Object>(std::move(o)); return v; }

    // --- dump ------------------------------------------------------------
    static void dump_string(std::ostream& os, const std::string& s) {
        os.put('"');
        for (unsigned char c : s) {
            switch (c) {
                case '"':  os << "\\\""; break;
                case '\\': os << "\\\\"; break;
                case '\b': os << "\\b"; break;
                case '\f': os << "\\f"; break;
                case '\n': os << "\\n"; break;
                case '\r': os << "\\r"; break;
                case '\t': os << "\\t"; break;
                default:
                    if (c < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                        os << buf;
                    } else {
                        os.put(static_cast<char>(c));
                    }
            }
        }
        os.put('"');
    }

    void dump_to(std::ostream& os) const {
        switch (type_) {
            case Type::Null:   os << "null"; break;
            case Type::Bool:   os << (bool_ ? "true" : "false"); break;
            case Type::Int:    os << int_; break;
            case Type::Double: {
                // %.17g gives a round-trippable representation for any
                // finite double. Append ".0" when %.17g produced a bare
                // integer-looking string ("1500" for 1.5e3), so re-parsing
                // preserves the Double type instead of promoting to Int.
                char buf[32];
                int len = std::snprintf(buf, sizeof(buf), "%.17g", dbl_);
                bool needs_dot = true;
                for (int i = 0; i < len; ++i) {
                    char c = buf[i];
                    if (c == '.' || c == 'e' || c == 'E' ||
                        c == 'n' || c == 'i') {  // nan / inf
                        needs_dot = false;
                        break;
                    }
                }
                os << buf;
                if (needs_dot) os << ".0";
                break;
            }
            case Type::String: dump_string(os, str_); break;
            case Type::Array: {
                // A moved-from Value keeps type_ but nulls arr_/obj_ — guard
                // like size()/empty()/as_array() rather than dereferencing.
                os.put('[');
                if (arr_) {
                    bool first = true;
                    for (const auto& e : *arr_) {
                        if (!first) os.put(',');
                        first = false;
                        e.dump_to(os);
                    }
                }
                os.put(']');
                break;
            }
            case Type::Object: {
                os.put('{');
                if (obj_) {
                    bool first = true;
                    for (const auto& kv : *obj_) {
                        if (!first) os.put(',');
                        first = false;
                        dump_string(os, kv.first);
                        os.put(':');
                        kv.second.dump_to(os);
                    }
                }
                os.put('}');
                break;
            }
        }
    }

    // --- parser ----------------------------------------------------------
    class Parser {
    public:
        explicit Parser(std::string_view s)
            : p_(s.data()), end_(s.data() + s.size()), line_start_(s.data()) {}

        [[nodiscard]] bool at_end() const { return p_ >= end_; }

        [[noreturn]] void fail(const char* msg) const {
            throw parse_error(msg, line_, static_cast<std::size_t>(p_ - line_start_) + 1);
        }

        void skip_ws() {
            for (;;) {
                // Fast path: run of whitespace / newlines. `\r`/`\n` bump the
                // line counter; other whitespace only advances the cursor.
                while (p_ < end_) {
                    char c = *p_;
                    if (c == ' ' || c == '\t')      { ++p_; continue; }
                    if (c == '\n')                  { newline(); continue; }
                    if (c == '\r')                  {
                        ++p_;
                        if (p_ < end_ && *p_ == '\n') { newline(); }
                        else { ++line_; line_start_ = p_; }
                        continue;
                    }
                    break;
                }
                if (end_ - p_ >= 2 && p_[0] == '/' && p_[1] == '/') {
                    p_ += 2;
                    while (p_ < end_ && *p_ != '\n') ++p_;
                    continue;
                }
                if (end_ - p_ >= 2 && p_[0] == '/' && p_[1] == '*') {
                    p_ += 2;
                    while (end_ - p_ >= 2 && !(p_[0] == '*' && p_[1] == '/')) {
                        if (*p_ == '\n') { newline(); }
                        else if (*p_ == '\r') {
                            ++p_;
                            if (p_ < end_ && *p_ == '\n') { newline(); }
                            else { ++line_; line_start_ = p_; }
                        }
                        else { ++p_; }
                    }
                    if (end_ - p_ < 2) fail("unterminated /* comment");
                    p_ += 2;
                    continue;
                }
                return;
            }
        }

        Value parse_value(int depth) {
            if (depth > kMaxDepth) fail("nesting too deep");
            skip_ws();
            if (at_end()) fail("unexpected end of input");
            char c = *p_;
            switch (c) {
                case '{': return parse_object(depth);
                case '[': return parse_array(depth);
                case '"': return Value::make_string(parse_string());
                case 't': return parse_literal("true", 4,  Value::make_bool(true));
                case 'f': return parse_literal("false", 5, Value::make_bool(false));
                case 'n': return parse_literal("null", 4,  Value{});
                default:
                    if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
                    fail("unexpected character");
            }
        }

    private:
        void newline() { ++p_; ++line_; line_start_ = p_; }

        Value parse_object(int depth) {
            ++p_;  // consume '{'
            Object out;
            skip_ws();
            if (!at_end() && *p_ == '}') { ++p_; return Value::make_object(std::move(out)); }
            for (;;) {
                skip_ws();
                if (at_end() || *p_ != '"') fail("expected string key");
                std::string key = parse_string();
                skip_ws();
                if (at_end() || *p_ != ':') fail("expected ':' after key");
                ++p_;
                out.emplace_back(std::move(key), parse_value(depth + 1));
                skip_ws();
                if (at_end()) fail("unterminated object");
                if (*p_ == ',') { ++p_; continue; }
                if (*p_ == '}') { ++p_; break; }
                fail("expected ',' or '}' in object");
            }
            return Value::make_object(std::move(out));
        }

        Value parse_array(int depth) {
            ++p_;  // consume '['
            Array out;
            skip_ws();
            if (!at_end() && *p_ == ']') { ++p_; return Value::make_array(std::move(out)); }
            for (;;) {
                out.push_back(parse_value(depth + 1));
                skip_ws();
                if (at_end()) fail("unterminated array");
                if (*p_ == ',') { ++p_; continue; }
                if (*p_ == ']') { ++p_; break; }
                fail("expected ',' or ']' in array");
            }
            return Value::make_array(std::move(out));
        }

        Value parse_literal(const char* word, std::size_t n, Value ret) {
            if (static_cast<std::size_t>(end_ - p_) < n ||
                std::memcmp(p_, word, n) != 0) {
                fail("bad literal");
            }
            p_ += n;
            return ret;
        }

        // Parse `[begin, end)` as a double using the C locale, so callers
        // that have set LC_NUMERIC=de_DE.UTF-8 (comma decimal) don't break
        // JSON parsing. strtod_l is a GNU/POSIX extension; the process-wide
        // C locale handle is cached in a static once-init.
        double parse_double_locale_safe(const char* begin, const char* end) {
            const std::size_t n = static_cast<std::size_t>(end - begin);
            char stack[64];
            std::string heap;
            char* buf;
            if (n < sizeof(stack)) {
                std::memcpy(stack, begin, n);
                stack[n] = '\0';
                buf = stack;
            } else {
                heap.assign(begin, n);
                buf = heap.data();
            }
            static ::locale_t c_loc = ::newlocale(LC_NUMERIC_MASK, "C", nullptr);
            errno = 0;
            char* endp = nullptr;
            // On the rare `newlocale` failure, fall back to plain strtod
            // rather than passing (locale_t)0 to strtod_l (UB). The process
            // locale is almost always "C" for LC_NUMERIC anyway.
            const double d = c_loc ? ::strtod_l(buf, &endp, c_loc)
                                   : std::strtod(buf, &endp);
            if (endp != buf + n) fail("bad number");
            // Only overflow is a hard error — gradual underflow (subnormals
            // like 1e-320) also sets ERANGE but yields a legitimate value.
            if (errno == ERANGE && (d == HUGE_VAL || d == -HUGE_VAL)) {
                fail("number out of range");
            }
            return d;
        }

        static int hex_val(char h) {
            if (h >= '0' && h <= '9') return h - '0';
            if (h >= 'a' && h <= 'f') return h - 'a' + 10;
            if (h >= 'A' && h <= 'F') return h - 'A' + 10;
            return -1;
        }

        unsigned parse_hex4() {
            if (end_ - p_ < 4) fail("bad \\u escape");
            unsigned cp = 0;
            for (int i = 0; i < 4; ++i) {
                int v = hex_val(*p_++);
                if (v < 0) fail("bad hex digit in \\u escape");
                cp = (cp << 4) | static_cast<unsigned>(v);
            }
            return cp;
        }

        static void encode_utf8(unsigned cp, std::string& out) {
            if (cp < 0x80) {
                out.push_back(static_cast<char>(cp));
            } else if (cp < 0x800) {
                out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else if (cp < 0x10000) {
                out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else {
                out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
        }

        std::string parse_string() {
            if (*p_ != '"') fail("expected '\"'");
            ++p_;
            // Fast path: scan for the first byte that would need special
            // handling ('"', '\\', or a control char). If the run is
            // escape-free we copy in one shot.
            std::string out;
            const char* run_begin = p_;
            while (p_ < end_) {
                unsigned char c = static_cast<unsigned char>(*p_);
                if (c == '"') {
                    out.append(run_begin, p_);
                    ++p_;
                    return out;
                }
                if (c == '\\') {
                    out.append(run_begin, p_);
                    ++p_;
                    if (at_end()) fail("dangling escape");
                    char esc = *p_++;
                    switch (esc) {
                        case '"':  out.push_back('"');  break;
                        case '\\': out.push_back('\\'); break;
                        case '/':  out.push_back('/');  break;
                        case 'b':  out.push_back('\b'); break;
                        case 'f':  out.push_back('\f'); break;
                        case 'n':  out.push_back('\n'); break;
                        case 'r':  out.push_back('\r'); break;
                        case 't':  out.push_back('\t'); break;
                        case 'u': {
                            unsigned cp = parse_hex4();
                            // High surrogate → look for the low surrogate.
                            if (cp >= 0xD800 && cp <= 0xDBFF) {
                                if (end_ - p_ >= 2 && p_[0] == '\\' && p_[1] == 'u') {
                                    p_ += 2;
                                    unsigned low = parse_hex4();
                                    if (low < 0xDC00 || low > 0xDFFF)
                                        fail("bad UTF-16 low surrogate");
                                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                                } else {
                                    fail("lone high surrogate");
                                }
                            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                                fail("lone low surrogate");
                            }
                            encode_utf8(cp, out);
                            break;
                        }
                        default: fail("bad string escape");
                    }
                    run_begin = p_;
                    continue;
                }
                // Control chars (including bare newline/tab) are rejected per
                // RFC 8259; must be escaped in JSON strings.
                if (c < 0x20) fail("control character in string");
                ++p_;
            }
            fail("unterminated string");
        }

        Value parse_number() {
            const char* start = p_;
            bool negative = false;
            if (*p_ == '-') { negative = true; ++p_; }
            if (at_end()) fail("bad number");

            // Integer part.
            const char* int_begin = p_;
            unsigned long long uval = 0;
            bool overflow = false;
            if (*p_ == '0') {
                ++p_;
                // Leading zero is only legal if standalone or before '.'/'e'.
            } else if (*p_ >= '1' && *p_ <= '9') {
                constexpr unsigned long long kUllMax =
                    std::numeric_limits<unsigned long long>::max();
                while (p_ < end_ && *p_ >= '0' && *p_ <= '9') {
                    unsigned d = static_cast<unsigned>(*p_ - '0');
                    if (uval > (kUllMax - d) / 10ULL) overflow = true;
                    uval = uval * 10ULL + d;
                    ++p_;
                }
            } else {
                fail("bad number");
            }
            if (p_ == int_begin) fail("bad number");

            bool is_float = false;
            if (p_ < end_ && *p_ == '.') {
                is_float = true; ++p_;
                if (p_ >= end_ || *p_ < '0' || *p_ > '9') fail("bad number");
                while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
            }
            if (p_ < end_ && (*p_ == 'e' || *p_ == 'E')) {
                is_float = true; ++p_;
                if (p_ < end_ && (*p_ == '+' || *p_ == '-')) ++p_;
                if (p_ >= end_ || *p_ < '0' || *p_ > '9') fail("bad number");
                while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
            }

            if (is_float) {
                const double d = parse_double_locale_safe(start, p_);
                return Value::make_double(d);
            }

            if (overflow) {
                // Integer path overflowed a 64-bit range — promote to double
                // rather than reject the file. Loses precision above 2^53.
                const double d = parse_double_locale_safe(start, p_);
                return Value::make_double(d);
            }

            constexpr unsigned long long kLlMaxPlusOne =
                static_cast<unsigned long long>(
                    std::numeric_limits<long long>::max()) + 1ULL;
            if (negative) {
                // Two's-complement range: |LLONG_MIN| == 2^63.
                if (uval > kLlMaxPlusOne) {
                    // Overflow of signed range — promote to double.
                    return Value::make_double(-static_cast<double>(uval));
                }
                if (uval == kLlMaxPlusOne) {
                    return Value::make_int(std::numeric_limits<long long>::min());
                }
                return Value::make_int(-static_cast<long long>(uval));
            }
            if (uval > static_cast<unsigned long long>(
                    std::numeric_limits<long long>::max())) {
                return Value::make_double(static_cast<double>(uval));
            }
            return Value::make_int(static_cast<long long>(uval));
        }

        const char* p_;
        const char* end_;
        const char* line_start_;
        std::size_t line_ = 1;
    };
};

// --- get<T> specializations -------------------------------------------------

template <> inline std::string Value::get<std::string>() const {
    if (!is_string()) throw std::runtime_error("mini_json: not a string");
    return str_;
}
template <> inline bool Value::get<bool>() const {
    if (!is_boolean()) throw std::runtime_error("mini_json: not a boolean");
    return bool_;
}
template <> inline long long Value::get<long long>() const {
    if (is_number_integer()) return int_;
    if (is_number_float())   return static_cast<long long>(dbl_);
    throw std::runtime_error("mini_json: not a number");
}
template <> inline int Value::get<int>() const {
    return static_cast<int>(get<long long>());
}
template <> inline double Value::get<double>() const {
    if (is_number_float())   return dbl_;
    if (is_number_integer()) return static_cast<double>(int_);
    throw std::runtime_error("mini_json: not a number");
}

}  // namespace ccl::selection::mini_json
