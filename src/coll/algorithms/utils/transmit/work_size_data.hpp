/*
 Copyright 2016-2020 Intel Corporation
 
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

#include "common/utils/rounding.hpp"

enum class AccessGranularity { Byte, Item, Vector };

template <typename message_t>
struct WorkSizeData {
    constexpr static size_t MAX_RANKS = 16; // TODO take this number from central source
    WorkSizeData(size_t nChunks, size_t size, size_t elemSize) {
        CCL_THROW_IF_NOT(nChunks <= MAX_RANKS, "too many offsets");

        size_t msgSize = divUp(size, sizeof(message_t));
        size_t chunkSize = divUp(msgSize, nChunks);

        // each rank in bytes
        size_t basicSize = ((size / sizeof(message_t)) / nChunks) * sizeof(message_t);
        // all ranks in bytes - size
        size_t remainder = size - basicSize * nChunks;

        this->workSize = chunkSize * sizeof(message_t);
        this->basicSize = basicSize;
        this->chunksFullSpillover = remainder / sizeof(message_t);
        this->lastChunkSpillover = remainder % sizeof(message_t);
        this->elemSize = elemSize;

        size_t tmp = 0;
        for (size_t i = 0; i < nChunks; ++i) {
            this->offsets[i] = tmp;
            this->nelems[i] = this->calcNelems(i, nChunks);
            tmp += this->nelems[i];
        }
    }

    inline size_t GetNElems(size_t i) {
        return nelems[i];
    }
    inline size_t GetOffset(size_t i) {
        return offsets[i];
    }
    // all in bytes
    size_t workSize;

private:
    size_t basicSize;
    size_t chunksFullSpillover;
    size_t lastChunkSpillover;
    size_t elemSize;
    size_t nelems[MAX_RANKS];
    size_t offsets[MAX_RANKS];

    size_t calcNelems(size_t chunkIdx, size_t nChunks) {
        size_t ret = basicSize;

        if (chunkIdx < chunksFullSpillover) {
            ret += sizeof(message_t);
        }
        // only last chunk handles spillover smaller than message_t
        //
        // sample: 35 bytes,
        // message_t: 16 bytes
        //
        // rank 0: 16 bytes
        // rank 1: 16 bytes
        // rank 2: 0 bytes
        // rank 3: 3 bytes
        //
        // this keeps all chunks aligned
        // if the 3 bytes were handled by rank 3, rank 4 would have unaligned data
        else if (chunkIdx == (nChunks - 1)) {
            ret += lastChunkSpillover;
        }
        return ret / elemSize;
    }
    size_t calcOffset(size_t chunkIdx, size_t nChunks) {
        size_t ret = 0;
        // TODO find better solution
        for (size_t i = 0; i < chunkIdx; ++i) {
            ret += this->calcNelems(i, nChunks);
        }
        return ret;
    }
};
