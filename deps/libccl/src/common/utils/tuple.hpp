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

#include <utility>
#include <tuple>

template <int...>
struct numeric_sequence {};

template <int current_number, int... final_number>
struct sequence_generator
        : sequence_generator<current_number - 1, current_number - 1, final_number...> {};

template <int... final_number>
struct sequence_generator<0, final_number...> {
    using type = numeric_sequence<final_number...>;
};

template <int CurIndex, class T, class U, class... Args>
struct get_tuple_elem_index {
    static constexpr int index = get_tuple_elem_index<CurIndex + 1, T, Args...>::index;
};

template <int CurIndex, class T, class... Args>
struct get_tuple_elem_index<CurIndex, T, T, Args...> {
    static constexpr int index = CurIndex;
};

template <class T, class... Args>
typename std::remove_reference<typename std::remove_cv<T>::type>::type& ccl_tuple_get(
    std::tuple<Args...>& t) {
    using non_cv_type = typename std::remove_cv<T>::type;
    using non_ref_type = typename std::remove_reference<non_cv_type>::type;
    return std::get<get_tuple_elem_index<0, non_ref_type, Args...>::index>(t);
}

template <class T, class... Args>
const typename std::remove_reference<typename std::remove_cv<T>::type>::type& ccl_tuple_get(
    const std::tuple<Args...>& t) {
    using non_cv_type = typename std::remove_cv<T>::type;
    using non_ref_type = typename std::remove_reference<non_cv_type>::type;
    return std::get<get_tuple_elem_index<0, non_ref_type, Args...>::index>(t);
}

template <class specific_tuple, class functor, size_t cur_index>
void ccl_tuple_for_each_impl(specific_tuple&& t, functor f, std::true_type tuple_finished) {
    // nothing to do
}

template <class specific_tuple, class functor, size_t cur_index>
void ccl_tuple_for_each_impl(specific_tuple&& t, functor f, std::false_type tuple_not_finished) {
    f(std::get<cur_index>(std::forward<specific_tuple>(t)));

    constexpr std::size_t tuple_size =
        std::tuple_size<typename std::remove_reference<specific_tuple>::type>::value;

    using is_tuple_finished_t = std::integral_constant<bool, cur_index + 1 >= tuple_size>;

    ccl_tuple_for_each_impl<specific_tuple, functor, cur_index + 1>(
        std::forward<specific_tuple>(t), f, is_tuple_finished_t{});
}

template <class specific_tuple, class functor, size_t cur_index = 0>
void ccl_tuple_for_each(specific_tuple&& t, functor&& f) {
    constexpr std::size_t tuple_size =
        std::tuple_size<typename std::remove_reference<specific_tuple>::type>::value;
    static_assert(tuple_size != 0, "Nothing to do, tuple is empty");

    using is_tuple_finished_t = std::integral_constant<bool, cur_index >= tuple_size>;
    ccl_tuple_for_each_impl<specific_tuple, functor, cur_index>(
        std::forward<specific_tuple>(t), f, is_tuple_finished_t{});
}

/* for each with args */
template <class specific_tuple, class functor, size_t cur_index, class... args_t>
void ccl_tuple_for_each_args_impl(specific_tuple&& t,
                                  functor f,
                                  std::true_type tuple_finished,
                                  args_t&&... args) {
    // nothing to do
}

template <class specific_tuple, class functor, size_t cur_index, class... args_t>
void ccl_tuple_for_each_args_impl(specific_tuple&& t,
                                  functor f,
                                  std::false_type tuple_not_finished,
                                  args_t&&... args) {
    f(std::get<cur_index>(std::forward<specific_tuple>(t)), std::forward<args_t>(args)...);

    constexpr std::size_t tuple_size =
        std::tuple_size<typename std::remove_reference<specific_tuple>::type>::value;

    using is_tuple_finished_t = std::integral_constant<bool, cur_index + 1 >= tuple_size>;

    ccl_tuple_for_each_args_impl<specific_tuple, functor, cur_index + 1>(
        std::forward<specific_tuple>(t), f, is_tuple_finished_t{}, std::forward<args_t>(args)...);
}

template <class specific_tuple, class functor, class... args_t, size_t cur_index = 0>
void ccl_tuple_for_each_args(specific_tuple&& t, functor&& f, args_t&&... args) {
    constexpr std::size_t tuple_size =
        std::tuple_size<typename std::remove_reference<specific_tuple>::type>::value;
    static_assert(tuple_size != 0, "Nothing to do, tuple is empty");

    using is_tuple_finished_t = std::integral_constant<bool, cur_index >= tuple_size>;
    ccl_tuple_for_each_args_impl<specific_tuple, functor, cur_index>(
        std::forward<specific_tuple>(t), f, is_tuple_finished_t{}, std::forward<args_t>(args)...);
}

template <typename specific_tuple, size_t cur_index, typename functor, class... FunctionArgs>
void ccl_tuple_for_each_indexed_impl(functor&,
                                     std::true_type tuple_finished,
                                     const FunctionArgs&... args) {}

template <typename specific_tuple, size_t cur_index, typename functor, class... FunctionArgs>
void ccl_tuple_for_each_indexed_impl(functor& f,
                                     std::false_type tuple_not_finished,
                                     const FunctionArgs&... args) {
    using tuple_element_t = typename std::tuple_element<cur_index, specific_tuple>::type;

    f.template invoke<cur_index, tuple_element_t>(args...);

    constexpr std::size_t tuple_size =
        std::tuple_size<typename std::remove_reference<specific_tuple>::type>::value;

    using is_tuple_finished_t = std::integral_constant<bool, cur_index + 1 >= tuple_size>;

    ccl_tuple_for_each_indexed_impl<specific_tuple, cur_index + 1, functor>(
        f, is_tuple_finished_t{}, args...);
}

template <typename specific_tuple, typename functor, class... FunctionArgs>
void ccl_tuple_for_each_indexed(functor& f, const FunctionArgs&... args) {
    constexpr std::size_t tuple_size =
        std::tuple_size<typename std::remove_reference<specific_tuple>::type>::value;
    static_assert(tuple_size != 0, "Nothing to do, tuple is empty");

    using is_tuple_finished_t = std::false_type; //non-empty tuple started
    ccl_tuple_for_each_indexed_impl<specific_tuple, 0, functor, FunctionArgs...>(
        f, is_tuple_finished_t{}, args...);
}

template <class T, class... Others>
struct is_one_of : std::false_type {};

template <class T, class Front, class... Others>
struct is_one_of<T, Front, Others...> : public std::conditional<std::is_same<T, Front>::value,
                                                                std::true_type,
                                                                is_one_of<T, Others...>>::type {};
