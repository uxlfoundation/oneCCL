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

#include "coll/selection/selector.hpp"
#include "common/utils/tuple.hpp"

#include <tuple>

template <ccl_coll_type... registered_coll_id>
class ccl_algorithm_selector_wrapper {
public:
    struct selector_init_functor {
        template <typename T>
        void operator()(T& t) const {
            t.init();
        }
    };

    struct selector_print_functor {
        template <typename T>
        void operator()(T& t) const {
            t.print();
        }
    };

    void init() {
        ccl_tuple_for_each(selectors, selector_init_functor());
    }

    void print() {
        ccl_tuple_for_each(selectors, selector_print_functor());
    }

    template <ccl_coll_type coll_id>
    typename ccl_algorithm_selector<coll_id>::type get(const ccl_selector_param& param) const {
        CCL_THROW_IF_NOT(
            coll_id == param.ctype, "expected coll_id ", coll_id, ", got ", param.ctype);
        return std::get<coll_id>(selectors).get(param);
    }

private:
    using algo_selectors = std::tuple<ccl_algorithm_selector<registered_coll_id>...>;
    algo_selectors selectors;
};
