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

#include "coll/selection/selector_helper.hpp"
#include "exec/exec.hpp"

#include <set>
#include <sstream>
#include <string>
#include <utility>

#define CCL_SELECTION_MAX_COLL_SIZE     (std::numeric_limits<size_t>::max())
#define CCL_SELECTION_MAX_COLL_SIZE_STR std::string("max")
#define CCL_SELECTION_BLOCK_DELIMETER   ';'
#define CCL_SELECTION_ALGO_DELIMETER    ':'
#define CCL_SELECTION_SIZE_DELIMETER    '-'

std::string to_string(const ccl_selector_param& param);

template <typename algo_group_type>
void ccl_selection_unpack_elem(size_t& size,
                               algo_group_type& algo,
                               ccl_selection_border_type& border,
                               const ccl_selection_table_iter_t<algo_group_type>& it,
                               const ccl_selection_table_t<algo_group_type>& table) {
    if (it != table.end()) {
        size = it->first;
        algo = it->second.first;
        border = it->second.second;
        LOG_TRACE("size ",
                  (size == CCL_SELECTION_MAX_COLL_SIZE) ? CCL_SELECTION_MAX_COLL_SIZE_STR
                                                        : std::to_string(size),
                  ", algo ",
                  ccl_coll_algorithm_to_str(algo),
                  ", border ",
                  border);
    }
}

template <typename algo_group_type>
void fill_table_from_str(const std::string& str_to_parse,
                         ccl_selection_table_t<algo_group_type>& table) {
    std::string block;
    std::string algo_name_str;
    std::string size_str;
    std::stringstream full_stream;
    std::stringstream block_stream;
    size_t left_size, right_size;

    /* format: <algo>:<size1-size2>;<algo>:<size1-size2>; ... */

    full_stream.str(str_to_parse);
    while (std::getline(full_stream, block, CCL_SELECTION_BLOCK_DELIMETER)) {
        if (block.empty()) {
            CCL_THROW("Empty block detected in string: ", str_to_parse);
        }
        LOG_TRACE(block);
        block_stream.str(block);
        try {
            if (!std::getline(block_stream, algo_name_str, CCL_SELECTION_ALGO_DELIMETER))
                CCL_THROW(
                    "can not parse algorithm name from string: ", str_to_parse, ", block: ", block);
        }
        catch (const std::istream::failure& e) {
            LOG_ERROR("exception happened: ",
                      e.what(),
                      "\nerror bits are:\nfailbit: ",
                      block_stream.fail(),
                      "\neofbit: ",
                      block_stream.eof(),
                      "\nbadbit: ",
                      block_stream.bad());
            CCL_THROW(
                "can not parse algorithm name from string: ", str_to_parse, ", block: ", block);
        }

        LOG_TRACE("block ", block, ", algo_name_str ", algo_name_str);

        algo_group_type algo = ccl_coll_algorithm_from_str<algo_group_type>(algo_name_str);

        if (algo_name_str.length() == block.length()) {
            /* set the single algorithm for the whole range */
            table.clear();
            ccl_algorithm_selector_base<algo_group_type>::insert(
                table, 0, CCL_SELECTION_MAX_COLL_SIZE, algo);
        }
        else {
            try {
                block_stream.str(block.substr(algo_name_str.length() + 1));
                if (!std::getline(block_stream, size_str, CCL_SELECTION_SIZE_DELIMETER))
                    CCL_THROW(
                        "can not parse left size from string: ", str_to_parse, ", block: ", block);
                if (!size_str.compare(CCL_SELECTION_MAX_COLL_SIZE_STR))
                    left_size = CCL_SELECTION_MAX_COLL_SIZE;
                else
                    left_size = std::strtoul(size_str.c_str(), nullptr, 10);
            }
            catch (const std::exception& e) {
                LOG_ERROR("exception happened during left size parsing: ", e.what());
                CCL_THROW(
                    "can not parse left size from string: ", str_to_parse, ", block: ", block);
            }

            try {
                block_stream.str(block.substr(algo_name_str.length() + size_str.length() + 2));
                if (!std::getline(block_stream, size_str, CCL_SELECTION_SIZE_DELIMETER))
                    CCL_THROW("can not parse second size from string: ",
                              str_to_parse,
                              ", block: ",
                              block);
                if (!size_str.compare(CCL_SELECTION_MAX_COLL_SIZE_STR))
                    right_size = CCL_SELECTION_MAX_COLL_SIZE;
                else
                    right_size = std::strtoul(size_str.c_str(), nullptr, 10);
            }
            catch (const std::exception& e) {
                LOG_ERROR("exception happened during right size parsing: ", e.what());
                CCL_THROW(
                    "can not parse right size from string: ", str_to_parse, ", block: ", block);
            }

            LOG_TRACE("algo ", algo_name_str, ", left ", left_size, ", right ", right_size);

            CCL_THROW_IF_NOT(left_size <= right_size,
                             "left border should be less or equal to right border (",
                             left_size,
                             ", ",
                             right_size,
                             ")");

            ccl_algorithm_selector_base<algo_group_type>::insert(
                table, left_size, right_size, algo);
        }
        block_stream.clear();
    }
}

template <typename algo_group_type>
void ccl_algorithm_selector_base<algo_group_type>::init() {
    const std::string& main_str_to_parse =
        ccl_algorithm_selector_helper<algo_group_type>::get_main_str_to_parse();
    const std::string& scaleout_str_to_parse =
        ccl_algorithm_selector_helper<algo_group_type>::get_scaleout_str_to_parse();

    size_t elem_size;
    algo_group_type elem_algo;
    ccl_selection_border_type elem_border;

    fill_table_from_str<algo_group_type>(main_str_to_parse, main_table);
    fill_table_from_str<algo_group_type>(scaleout_str_to_parse, scaleout_table);

    auto tables_to_check = std::vector<const ccl_selection_table_t<algo_group_type>*>{
        &main_table, &fallback_table, &scaleout_table
    };

    for (const auto& table : tables_to_check) {
        CCL_THROW_IF_NOT(table->size() >= 2, "selection table should have at least 2 entries");

        /* ensure that table has entries with size_t::max as key, i.e. able to cover all message sizes */
        CCL_THROW_IF_NOT(table->find(0) != table->end() &&
                             table->find(CCL_SELECTION_MAX_COLL_SIZE) != table->end(),
                         "selection table should have entries for min and max message sizes");

        /* check that table has expected left/right/both borders */
        std::set<ccl_selection_border_type> expected_left_and_both{ ccl_selection_border_left,
                                                                    ccl_selection_border_both };
        std::set<ccl_selection_border_type> expected_right{ ccl_selection_border_right };

        std::set<ccl_selection_border_type> expected_set = expected_left_and_both;
        for (const auto& elem : *table) {
            elem_size = elem.first;
            elem_algo = elem.second.first;
            elem_border = elem.second.second;

            if (expected_set.find(elem_border) == expected_set.end()) {
                print();
                CCL_THROW("unexpected elem in table: size ",
                          elem_size,
                          ", algo ",
                          ccl_coll_algorithm_to_str(elem_algo),
                          ", border_type ",
                          elem_border);
            }
            if (elem_border == ccl_selection_border_left)
                expected_set = expected_right;
            else if (elem_border == ccl_selection_border_right)
                expected_set = expected_left_and_both;
            else if (elem_border == ccl_selection_border_both)
                expected_set = expected_left_and_both;
        }
    }
}

template <typename algo_group_type>
void ccl_algorithm_selector_base<algo_group_type>::print() const {
    std::stringstream str;
    auto tables_to_print = std::vector<const ccl_selection_table_t<algo_group_type>*>{
        &main_table, &fallback_table, &scaleout_table
    };

    str << std::endl
        << ccl_coll_type_to_str(ccl_algorithm_selector_helper<algo_group_type>::get_coll_id())
        << " selection" << std::endl;

    for (const auto& table : tables_to_print) {
        std::string table_name = (table == &main_table) ? "main table" : "fallback table";
        table_name = (table == &scaleout_table) ? "scaleout table" : table_name;

        str << "  " << table_name << std::endl;
        str << ccl_algorithm_selector_base<algo_group_type>::table_to_str(*table);
    }
    LOG_DEBUG(str.str());
}

template <typename algo_group_type>
std::string ccl_algorithm_selector_base<algo_group_type>::table_to_str(
    const ccl_selection_table_t<algo_group_type>& table) {
    size_t elem_size;
    algo_group_type elem_algo;
    // elem_border represents which border the current is on.
    // ccl_selection_border_left means the current element is on the left border,
    // ccl_selection_border_right means the current element is on the right,
    // while ccl_selection_border_both means the current element covers all sizes.
    ccl_selection_border_type elem_border;

    std::stringstream str;
    for (auto it = table.begin(); it != table.end(); ++it) {
        ccl_selection_unpack_elem(elem_size, elem_algo, elem_border, it, table);

        size_t left_size = 0, right_size = 0;
        if (elem_border == ccl_selection_border_both) {
            left_size = right_size = elem_size;
        }
        else if (elem_border == ccl_selection_border_left) {
            left_size = elem_size;
            it++;
            if (it == table.end()) {
                CCL_THROW("missing right border");
            }
            ccl_selection_unpack_elem(elem_size, elem_algo, elem_border, it, table);
            CCL_THROW_IF_NOT(elem_border == ccl_selection_border_right);
            right_size = elem_size;
        }

        str << "    ["
            << ((left_size == CCL_SELECTION_MAX_COLL_SIZE) ? CCL_SELECTION_MAX_COLL_SIZE_STR
                                                           : std::to_string(left_size))
            << " - "
            << ((right_size == CCL_SELECTION_MAX_COLL_SIZE) ? CCL_SELECTION_MAX_COLL_SIZE_STR
                                                            : std::to_string(right_size))
            << "]: " << ccl_coll_algorithm_to_str(elem_algo) << std::endl;
    }
    return str.str();
}

// return algo in the table based on size in bytes
template <typename algo_group_type>
algo_group_type ccl_algorithm_selector_base<algo_group_type>::get_value_from_table(
    size_t size,
    const ccl_selection_table_t<algo_group_type>& table) {
    auto lower_bound = table.lower_bound(size);
    if (lower_bound == table.end()) {
        CCL_THROW("Size ", size, " is out of range. Check table configuration.");
    }
    size_t elem_size;
    algo_group_type elem_algo;
    ccl_selection_border_type elem_border;
    ccl_selection_unpack_elem(elem_size, elem_algo, elem_border, lower_bound, table);
    return elem_algo;
}

template <typename algo_group_type>
algo_group_type ccl_algorithm_selector_base<algo_group_type>::get(
    const ccl_selector_param& param) const {
    size_t elem_size;
    algo_group_type elem_algo;
    ccl_selection_border_type elem_border;

    LOG_DEBUG("param: ", ::to_string(param));

    size_t count = ccl_algorithm_selector_helper<algo_group_type>::get_count(param);

    if (param.hint_algo.has_value()) {
        elem_algo = static_cast<algo_group_type>(param.hint_algo.value);
        if (!ccl_algorithm_selector_helper<algo_group_type>::can_use(
                elem_algo, param, main_table)) {
            LOG_DEBUG("can not select hint algorithm: coll ",
                      ccl_coll_type_to_str(param.ctype),
                      ", count ",
                      count,
                      ", algo ",
                      ccl_coll_algorithm_to_str(elem_algo),
                      ", switch to regular selection");
        }
        else {
            LOG_DEBUG("selected hint algo: coll ",
                      ccl_coll_type_to_str(param.ctype),
                      ", count ",
                      count,
                      ", algo ",
                      ccl_coll_algorithm_to_str(elem_algo));
            return elem_algo;
        }
    }

    size_t size = count * param.dtype.size();

    // Firstly check the scale-out case
    if (param.is_scaleout) {
        auto lower_bound = scaleout_table.lower_bound(size);
        ccl_selection_unpack_elem(elem_size, elem_algo, elem_border, lower_bound, scaleout_table);

        if (lower_bound != scaleout_table.end() &&
            ccl_algorithm_selector_helper<algo_group_type>::can_use(
                elem_algo, param, scaleout_table)) {
            LOG_DEBUG("selected scale-out algo: coll ",
                      ccl_coll_type_to_str(param.ctype),
                      ", count ",
                      count,
                      ", algo ",
                      ccl_coll_algorithm_to_str(elem_algo));
            return elem_algo;
        }
    }

    auto lower_bound = main_table.lower_bound(size);
    ccl_selection_unpack_elem(elem_size, elem_algo, elem_border, lower_bound, main_table);

    if (lower_bound == main_table.end() ||
        !ccl_algorithm_selector_helper<algo_group_type>::can_use(elem_algo, param, main_table)) {
        CCL_THROW_IF_NOT(ccl::global_data::env().enable_algo_fallback,
                         "can not select algo from main table and fallback is disabled",
                         ", coll ",
                         ccl_coll_type_to_str(param.ctype),
                         ", count ",
                         count);

        lower_bound = fallback_table.lower_bound(size);
        ccl_selection_unpack_elem(elem_size, elem_algo, elem_border, lower_bound, fallback_table);
        CCL_THROW_IF_NOT(lower_bound != fallback_table.end(),
                         "can not select algorithm: coll ",
                         ccl_coll_type_to_str(param.ctype),
                         ", count ",
                         count);
        CCL_THROW_IF_NOT(ccl_algorithm_selector_helper<algo_group_type>::can_use(
                             elem_algo, param, fallback_table),
                         "can not select algorithm in fallback_table: coll ",
                         ccl_coll_type_to_str(param.ctype));
    }

    LOG_DEBUG("selected algo: coll ",
              ccl_coll_type_to_str(param.ctype),
              ", count ",
              count,
              ", algo ",
              ccl_coll_algorithm_to_str(elem_algo));

    return elem_algo;
}

template <typename algo_group_type>
void ccl_algorithm_selector_base<algo_group_type>::insert(
    ccl_selection_table_t<algo_group_type>& table,
    size_t left,
    size_t right,
    algo_group_type algo) {
    LOG_TRACE("left ",
              (left == CCL_SELECTION_MAX_COLL_SIZE) ? CCL_SELECTION_MAX_COLL_SIZE_STR
                                                    : std::to_string(left),
              ", right ",
              (right == CCL_SELECTION_MAX_COLL_SIZE) ? CCL_SELECTION_MAX_COLL_SIZE_STR
                                                     : std::to_string(right),
              ", algo ",
              ccl_coll_algorithm_to_str(algo));

    std::stringstream str;
    size_t elem_size, next_elem_size, prev_elem_size;
    algo_group_type elem_algo, next_elem_algo, prev_elem_algo;
    ccl_selection_border_type elem_border, next_elem_border, prev_elem_border;

    /* remove internal ranges between left and right */
    auto iter = table.lower_bound(left);
    ccl_selection_unpack_elem(elem_size, elem_algo, elem_border, iter, table);

    while (iter != table.end() && elem_size <= right) {
        ccl_selection_unpack_elem(elem_size, elem_algo, elem_border, iter, table);
        if (elem_border == ccl_selection_border_right) {
            iter++;
        }
        else {
            if (elem_border == ccl_selection_border_both) {
                iter = table.erase(iter);
            }
            else {
                auto next_iter = std::next(iter);
                ccl_selection_unpack_elem(
                    next_elem_size, next_elem_algo, next_elem_border, next_iter, table);
                CCL_THROW_IF_NOT(next_iter != table.end());
                CCL_THROW_IF_NOT(elem_border == ccl_selection_border_left);
                CCL_THROW_IF_NOT(next_elem_border == ccl_selection_border_right);
                CCL_THROW_IF_NOT(elem_algo == next_elem_algo);

                if (next_elem_size <= right) {
                    table.erase(iter);
                    iter = table.erase(next_iter);
                }
                else
                    iter++;
            }
        }
    }

    /* now it is safe to adjust remaining ranges */
    iter = table.lower_bound(left);
    while (iter != table.end()) {
        ccl_selection_unpack_elem(elem_size, elem_algo, elem_border, iter, table);

        CCL_THROW_IF_NOT(elem_border != ccl_selection_border_both);

        if (elem_size >= left && elem_size <= right) {
            if (elem_border == ccl_selection_border_right) {
                CCL_THROW_IF_NOT(left > 0);
                auto prev_iter = table.find(left - 1);
                ccl_selection_unpack_elem(
                    prev_elem_size, prev_elem_algo, prev_elem_border, prev_iter, table);
                if (prev_iter != table.end()) {
                    CCL_THROW_IF_NOT(prev_elem_border == ccl_selection_border_left);
                    prev_iter->second.second = ccl_selection_border_both; // set prev_elem_border
                }
                else {
                    table[left - 1] = iter->second;
                }
            }
            else if (elem_border == ccl_selection_border_left) {
                CCL_THROW_IF_NOT(right < CCL_SELECTION_MAX_COLL_SIZE);
                auto next_iter = table.find(right + 1);
                ccl_selection_unpack_elem(
                    next_elem_size, next_elem_algo, next_elem_border, next_iter, table);
                if (next_iter != table.end()) {
                    CCL_THROW_IF_NOT(next_elem_border == ccl_selection_border_right);
                    next_iter->second.second = ccl_selection_border_both; // set next_elem_border
                    table.erase(iter);
                }
                else {
                    table[right + 1] = iter->second;
                }
            }
            table.erase(iter);
            iter = table.lower_bound(left);
        }
        else {
            /* new range will be fully inside wider range or will not overlap with other ranges */
            if (elem_border == ccl_selection_border_right) {
                auto prev_iter = std::prev(iter);
                ccl_selection_unpack_elem(
                    prev_elem_size, prev_elem_algo, prev_elem_border, prev_iter, table);

                CCL_THROW_IF_NOT(right < CCL_SELECTION_MAX_COLL_SIZE);
                if (elem_size == right + 1) {
                    table[right + 1] = std::make_pair(elem_algo, ccl_selection_border_both);
                }
                else {
                    table[right + 1] = std::make_pair(elem_algo, ccl_selection_border_left);
                }

                CCL_THROW_IF_NOT(prev_iter != table.end());
                CCL_THROW_IF_NOT(prev_elem_border == ccl_selection_border_left);
                CCL_THROW_IF_NOT(elem_algo == prev_elem_algo);

                if (prev_elem_size == left - 1) {
                    table[left - 1] = std::make_pair(prev_elem_algo, ccl_selection_border_both);
                }
                else {
                    table[left - 1] = std::make_pair(prev_elem_algo, ccl_selection_border_right);
                }
            }
            else {
                /* do nothing */
            }
            iter = table.end();
        }
    }

    /* now it is safe to add left and right borders */
    CCL_THROW_IF_NOT(table.find(left) == table.end());
    CCL_THROW_IF_NOT(table.find(right) == table.end());

    if (left == right)
        table[left] = std::make_pair(algo, ccl_selection_border_both);
    else {
        table[left] = std::make_pair(algo, ccl_selection_border_left);
        table[right] = std::make_pair(algo, ccl_selection_border_right);
    }

    if (table.size() == 1)
        return;

    /* merge adjacent ranges for the same algorithm */
    for (iter = table.begin(); iter != table.end();) {
        ccl_selection_unpack_elem(elem_size, elem_algo, elem_border, iter, table);

        if (elem_border == ccl_selection_border_right || elem_border == ccl_selection_border_both) {
            auto next_iter = std::next(iter);
            if (next_iter == table.end()) {
                break;
            }

            ccl_selection_unpack_elem(
                next_elem_size, next_elem_algo, next_elem_border, next_iter, table);

            CCL_THROW_IF_NOT(next_elem_border == ccl_selection_border_left ||
                             next_elem_border == ccl_selection_border_both);

            if ((elem_size + 1) == next_elem_size && elem_algo == next_elem_algo) {
                /* do merge */
                if (elem_border == ccl_selection_border_both) {
                    iter->second.second = ccl_selection_border_left;

                    if (next_elem_border == ccl_selection_border_both) {
                        next_iter->second.second = ccl_selection_border_right;
                        iter++;
                    }
                    else if (next_elem_border == ccl_selection_border_left) {
                        iter = table.erase(next_iter);
                    }
                }
                else if (elem_border == ccl_selection_border_right) {
                    if (next_elem_border == ccl_selection_border_both) {
                        next_iter->second.second = ccl_selection_border_right;
                        iter = table.erase(iter);
                    }
                    else if (next_elem_border == ccl_selection_border_left) {
                        table.erase(next_iter);
                        iter = table.erase(iter);
                    }
                }
            }
            else {
                iter++;
            }
        }
        else {
            iter++;
        }
    }
}
