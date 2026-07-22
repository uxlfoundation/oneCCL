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

#include <ostream>
#include <stddef.h>
#include <sys/types.h>

class ccl_bin_tree {
public:
    ccl_bin_tree(int comm_size, int rank, bool is_main = true)
            : comm_size(comm_size),
              rank(rank),
              is_main(is_main) {
        calc_height(is_main);

        if (rank == default_root) {
            p = -1;
            l = -1;
            if (is_main) {
                r = height > 0 ? 1 << (height - 1) : -1;
            }
            else {
                if (comm_size == 1 << height) {
                    r = height > 0 ? (1 << height) - 1 : -1;
                }
                else {
                    r = height > 0 ? (1 << (height - 1)) - 1 : -1;
                }
            }
            return;
        }

        calc_parent();
        if (height > 0) {
            calc_left();
            calc_right();
        }
    }

    ccl_bin_tree(const ccl_bin_tree& other) = default;
    ccl_bin_tree& operator=(const ccl_bin_tree& other) = default;

    int left() const {
        return l;
    }

    int right() const {
        return r;
    }

    int parent() const {
        return p;
    }

    ccl_bin_tree copy_with_new_root(int new_root) const {
        ccl_bin_tree copy(*this);
        int root = static_cast<int>(new_root);

        //if current node will become a new root or node was a default root - the tree must be reconstruced
        if (copy.rank == root || copy.rank == default_root) {
            //create part of tree with the default root
            copy = ccl_bin_tree(static_cast<int>(comm_size),
                                copy.rank == default_root ? root : default_root,
                                is_main);
            copy.rank = root;
        }

        //swap default root with new root in any of left/right/parent nodes
        copy.reset_connections(root);

        return copy;
    }

    friend std::ostream& operator<<(std::ostream& str, const ccl_bin_tree& tree) {
        str << "parent " << tree.p << " -> rank " << tree.rank << " -> [left " << tree.l
            << ", right " << tree.r << "]";
        return str;
    }

private:
    void reset_connections(int new_root) {
        swap_if_any_of(p, default_root, new_root);
        swap_if_any_of(l, default_root, new_root);
        swap_if_any_of(r, default_root, new_root);
    }

    static void swap_if_any_of(int& node, int val1, int val2) {
        if (node == val1) {
            node = val2;
        }
        else if (node == val2) {
            node = val1;
        }
    }

    void calc_height(bool main_tree) {
        if (main_tree || rank == default_root) {
            while ((rank & (1 << height)) == 0 && (1 << height) < comm_size) {
                ++height;
            }
        }
        else {
            while ((rank & (1 << height)) != 0 && (1 << height) < comm_size) {
                ++height;
            }
        }
    }

    void calc_parent() {
        //find a parent using height, assume that rank is a right child
        int possible_parent_as_left = rank + (1 << height);
        //right child has a bit `1` a the position `height + 1` due to it is calculated as `parent + 2^(heightP-1)`
        //where heightP is parent's height i.e height + 1

        if ((rank & (1 << (height + 1))) ||
            //parent of the left rank is always bigger than its parent, check that we do not exceed comm size
            possible_parent_as_left > comm_size - 1) {
            //this is right child
            p = rank - (1 << height);
            if (p < 0) {
                p = 0;
            }
        }
        else {
            p = possible_parent_as_left;
        }
    }

    void calc_left() {
        l = rank - (1 << (height - 1));
        if (l <= 0) {
            l = -1;
        }
    }

    void calc_right() {
        r = rank + (1 << (height - 1));
        int limit = comm_size - 1;

        if (r > limit) {
            auto height_tmp = height;
            //need to decrease height to find most suitable right -- topmost right leaf case
            do {
                --height_tmp;
                if (height_tmp == 0) {
                    r = -1;
                    break;
                }
                r = rank + (1 << (height_tmp - 1));

            } while (r > limit);
        }
    }

    int comm_size;
    int rank;
    int height = 0;
    int p = -1;
    int l = -1;
    int r = -1;
    bool is_main;

    static const int default_root = 0;
};

class ccl_double_tree {
public:
    ccl_double_tree(int comm_size, int rank)
            : t1(comm_size, rank, true),
              t2(comm_size, rank, false) {
        //LOG_DEBUG("T1: ", t1);
        //LOG_DEBUG("T2: ", t2);
    }

    /**
     * Binary tree which consists of the current rank as a node and possible parent, left and right children.
     * Even ranks numbers are always inner nodes, odd rank numbers are always leaves
     * @return binary tree t1
     */
    const ccl_bin_tree& T1() const {
        return t1;
    }

    /**
     * Binary tree which consists of the current rank as a node and possible parent, left and right children.
     * Even ranks numbers are always leaves, odd rank numbers are always inner nodes
     * @return binary tree t2
     */
    const ccl_bin_tree& T2() const {
        return t2;
    }

    ccl_double_tree copy_with_new_root(int new_root) const {
        return ccl_double_tree(t1.copy_with_new_root(new_root), t2.copy_with_new_root(new_root));
    }

private:
    ccl_double_tree(ccl_bin_tree t1, ccl_bin_tree t2) : t1(t1), t2(t2) {}

    ccl_bin_tree t1;
    ccl_bin_tree t2;
};
