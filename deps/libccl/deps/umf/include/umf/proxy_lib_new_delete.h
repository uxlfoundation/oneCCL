/* ----------------------------------------------------------------------------
Copyright (c) 2018-2020 Microsoft Research, Daan Leijen
Copyright (C) 2024 Intel Corporation

This is free software; you can redistribute it and/or modify it under the
terms of the MIT license:

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

-----------------------------------------------------------------------------*/

#ifndef UMF_PROXY_LIB_NEW_DELETE_H
#define UMF_PROXY_LIB_NEW_DELETE_H

// ----------------------------------------------------------------------------
// This header provides convenient overrides for the new and
// delete operations in C++.
//
// This header should be included in only one source file!
//
// On Windows, or when linking dynamically with UMF, these
// can be more performant than the standard new-delete operations.
// See <https://en.cppreference.com/w/cpp/memory/new/operator_new>
// ---------------------------------------------------------------------------

#if defined(__cplusplus)
#include <new>

#ifndef _WIN32
#include <stdlib.h>
#endif // _WIN32

// disable warning 28251: "Inconsistent annotation for 'new': this instance
// has no annotations." because different Win SDKs use slightly different
// definitions of new
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 28251)
#endif // _MSC_VER

static inline void *internal_aligned_alloc(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return aligned_alloc(alignment, size);
#endif // _WIN32
}

#if defined(_MSC_VER) && defined(_Ret_notnull_) &&                             \
    defined(_Post_writable_byte_size_)
// stay consistent with VCRT definitions
#define decl_new(n) [[nodiscard]] _Ret_notnull_ _Post_writable_byte_size_(n)
#define decl_new_nothrow(n)                                                    \
    [[nodiscard]] _Ret_maybenull_ _Success_(return != NULL)                    \
        _Post_writable_byte_size_(n)
#else
#define decl_new(n) [[nodiscard]]
#define decl_new_nothrow(n) [[nodiscard]]
#endif // defined(_MSC_VER) && defined(_Ret_notnull_) && defined(_Post_writable_byte_size_)

void operator delete(void *p) noexcept { free(p); }
void operator delete[](void *p) noexcept { free(p); }

void operator delete(void *p, const std::nothrow_t &) noexcept { free(p); }
void operator delete[](void *p, const std::nothrow_t &) noexcept { free(p); }

decl_new(n) void *operator new(std::size_t n) noexcept(false) {
    void *ptr = malloc(n);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}
decl_new(n) void *operator new[](std::size_t n) noexcept(false) {
    void *ptr = malloc(n);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

decl_new_nothrow(n) void *operator new(std::size_t n,
                                       const std::nothrow_t &tag) noexcept {
    (void)(tag);
    return malloc(n);
}
decl_new_nothrow(n) void *operator new[](std::size_t n,
                                         const std::nothrow_t &tag) noexcept {
    (void)(tag);
    return malloc(n);
}

#if (__cplusplus >= 201402L || _MSC_VER >= 1916)
void operator delete(void *p, std::size_t n) noexcept {
    (void)(n);
    free(p);
}
void operator delete[](void *p, std::size_t n) noexcept {
    (void)(n);
    free(p);
}
#endif // (__cplusplus >= 201402L || _MSC_VER >= 1916)

#if (__cplusplus > 201402L || defined(__cpp_aligned_new))
void operator delete(void *p, std::align_val_t al) noexcept {
    (void)(al);
    free(p);
}
void operator delete[](void *p, std::align_val_t al) noexcept {
    (void)(al);
    free(p);
}
void operator delete(void *p, std::size_t n, std::align_val_t al) noexcept {
    (void)(n);
    (void)(al);
    free(p);
}
void operator delete[](void *p, std::size_t n, std::align_val_t al) noexcept {
    (void)(n);
    (void)(al);
    free(p);
}
void operator delete(void *p, std::align_val_t al,
                     const std::nothrow_t &) noexcept {
    (void)(al);
    free(p);
}
void operator delete[](void *p, std::align_val_t al,
                       const std::nothrow_t &) noexcept {
    (void)(al);
    free(p);
}

void *operator new(std::size_t n, std::align_val_t al) noexcept(false) {
    void *ptr = internal_aligned_alloc(static_cast<size_t>(al), n);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}
void *operator new[](std::size_t n, std::align_val_t al) noexcept(false) {
    void *ptr = internal_aligned_alloc(static_cast<size_t>(al), n);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}
void *operator new(std::size_t n, std::align_val_t al,
                   const std::nothrow_t &) noexcept {
    return internal_aligned_alloc(static_cast<size_t>(al), n);
}
void *operator new[](std::size_t n, std::align_val_t al,
                     const std::nothrow_t &) noexcept {
    return internal_aligned_alloc(static_cast<size_t>(al), n);
}

#if defined(_MSC_VER)
#pragma warning(pop)
#endif // _MSC_VER

#endif // (__cplusplus > 201402L || defined(__cpp_aligned_new))
#endif // defined(__cplusplus)

#endif // UMF_PROXY_LIB_NEW_DELETE_H
