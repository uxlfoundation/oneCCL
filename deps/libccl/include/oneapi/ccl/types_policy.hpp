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

#include <memory>

namespace ccl {
template <class impl_t>
class non_copyable {
public:
    non_copyable(const non_copyable&) = delete;
    non_copyable& operator=(const non_copyable& src) = delete;
    impl_t& operator=(const impl_t&) = delete;

protected:
    non_copyable() = default;
    ~non_copyable() = default;
};

template <class impl_t>
class non_movable {
public:
    non_movable(non_movable&&) = delete;
    non_movable& operator=(non_movable&& src) = delete;
    impl_t& operator=(impl_t&&) = delete;

protected:
    non_movable() = default;
    ~non_movable() = default;
};

template <class derived_t, class impl_t>
class pointer_on_impl {
protected:
    using impl_value_t = std::unique_ptr<impl_t>;
    using parent_t = derived_t;

    pointer_on_impl(impl_value_t&& impl) : pimpl(std::move(impl)) {}
    ~pointer_on_impl() = default;

    impl_t& get_impl() {
        return *pimpl;
    }
    const impl_t& get_impl() const {
        return *pimpl;
    }

private:
    impl_value_t pimpl;
};

template <class T>
struct copy_on_write_access_policy {
    using impl_t = T;
    using self_t = copy_on_write_access_policy<T>;

    template <class ccl_api_t>
    static void create(ccl_api_t* dst, const ccl_api_t& src) {
        static_assert(std::is_same<typename ccl_api_t::acc_policy_t, self_t>::value,
                      "ccl_api_t is not provide 'copy_on_write_access_policy'");
        if (dst != &src) {
            dst->get_impl().reset(new T(*src.get_impl().get()));
        }
    }

    template <class ccl_api_t>
    static void create(ccl_api_t* dst, ccl_api_t&& src) {
        static_assert(std::is_same<typename ccl_api_t::acc_policy_t, self_t>::value,
                      "ccl_api_t is not provide 'copy_on_write_access_policy'");
        if (dst != &src) {
            dst->get_impl() = std::move(src.get_impl());
        }
    }

    template <template <class...> class wrapper>
    static wrapper<T>& get_access(wrapper<T>& obj) {
        if (obj) {
            wrapper<T> copy{ new T(*obj) };
            obj.swap(copy);
        }
        return obj;
    }

    template <template <class...> class wrapper>
    static const wrapper<T>& get_access(const wrapper<T>& obj) {
        return obj;
    }
};

template <class T>
struct direct_access_policy {
    using impl_t = T;
    using self_t = direct_access_policy<T>;

    template <class ccl_api_t>
    static void create(ccl_api_t* dst, const ccl_api_t& src) {
        static_assert(std::is_same<typename ccl_api_t::acc_policy_t, self_t>::value,
                      "ccl_api_t is not provide 'copy_on_write_access_policy'");
        if (dst != &src) {
            dst->get_impl() = src.get_impl();
        }
    }

    template <class ccl_api_t>
    static void create(ccl_api_t* dst, ccl_api_t&& src) {
        static_assert(std::is_same<typename ccl_api_t::acc_policy_t, self_t>::value,
                      "ccl_api_t is not provide 'copy_on_write_access_policy'");
        if (dst != &src) {
            dst->get_impl() = std::move(src.get_impl());
        }
    }

    template <template <class...> class wrapper>
    static wrapper<T>& get_access(wrapper<T>& obj) {
        return obj;
    }

    template <template <class...> class wrapper>
    static const wrapper<T>& get_access(const wrapper<T>& obj) {
        return obj;
    }
};

template <class derived_t,
          template <class>
          class access_policy_t,
          class impl_t,
          template <class...> class pointer_t = std::shared_ptr>
class ccl_api_base_copyable : protected access_policy_t<impl_t> {
protected:
    using impl_value_t = pointer_t<impl_t>;
    using parent_t = derived_t;
    using acc_policy_t = access_policy_t<impl_t>;

    friend struct access_policy_t<impl_t>;

    ccl_api_base_copyable(impl_value_t&& impl) : pimpl(std::move(impl)) {}

    ccl_api_base_copyable(const ccl_api_base_copyable& src) {
        access_policy_t<impl_t>::create(this, src);
    }
    ccl_api_base_copyable(ccl_api_base_copyable&& src) {
        access_policy_t<impl_t>::create(this, std::move(src));
    }
    ccl_api_base_copyable& operator=(const ccl_api_base_copyable& src) = delete;
    ccl_api_base_copyable& operator=(ccl_api_base_copyable&& src) = delete;
    ~ccl_api_base_copyable() = default;

    impl_value_t& get_impl() {
        return (access_policy_t<impl_t>::get_access(pimpl));
    }

    const impl_value_t& get_impl() const {
        return (access_policy_t<impl_t>::get_access(pimpl));
    }

private:
    impl_value_t pimpl;
};

template <class derived_t,
          template <class>
          class access_policy_t,
          class impl_t,
          template <class...> class pointer_t = std::unique_ptr>
class ccl_api_base_movable : protected access_policy_t<impl_t> {
protected:
    using impl_value_t = pointer_t<impl_t>;
    using parent_t = derived_t;
    using acc_policy_t = access_policy_t<impl_t>;

    friend struct access_policy_t<impl_t>;

    ccl_api_base_movable(impl_value_t&& impl) : pimpl(std::move(impl)) {}

    ccl_api_base_movable(ccl_api_base_movable&& src) {
        access_policy_t<impl_t>::create(this, std::move(src));
    }
    ccl_api_base_movable& operator=(ccl_api_base_movable&& src) = delete;
    ccl_api_base_movable(const ccl_api_base_movable& src) = delete;
    ccl_api_base_movable& operator=(const ccl_api_base_movable& src) = delete;
    ~ccl_api_base_movable() = default;

    impl_value_t& get_impl() {
        return (access_policy_t<impl_t>::get_access(pimpl));
    }

    const impl_value_t& get_impl() const {
        return (access_policy_t<impl_t>::get_access(pimpl));
    }

private:
    impl_value_t pimpl;
};

namespace detail {
template <class attr, attr id>
struct ccl_api_type_attr_traits {};

template <class attr_id_type, attr_id_type attr_id, class value_type>
struct attr_value_triple {
    using type_t = attr_id_type;
    using value_t = value_type;
    static constexpr attr_id_type idx() {
        return attr_id;
    }

    explicit attr_value_triple(value_t val) : m_val(val) {}
    const value_type& val() {
        return m_val;
    }

private:
    value_t m_val;
};
} // namespace detail

} // namespace ccl
