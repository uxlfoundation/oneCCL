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

/*
 * Base16 encode/decode copied from Sandia OpenSHMEM
 * https://github.com/Sandia-OpenSHMEM
 *
 * See LICENSE-SandiaOpenSHMEM in top-level directory.
 */

#ifndef PMI_RT_CODEC_H
#define PMI_RT_CODEC_H

#include <string.h>
#include <stdio.h>

static inline int encode(const void *inval, int invallen, char *outval, int outvallen) {
    static unsigned char encodings[] = { '0', '1', '2', '3', '4', '5', '6', '7',
                                         '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };
    int i;

    if (invallen * 2 + 1 > outvallen)
        return -1;

    for (i = 0; i < invallen; i++) {
        outval[2 * i] = encodings[((unsigned char *)inval)[i] & 0xf];
        outval[2 * i + 1] = encodings[((unsigned char *)inval)[i] >> 4];
    }

    outval[invallen * 2] = '\0';

    return 0;
}

static inline int decode(const char *inval, void *outval, int outvallen) {
    int i;
    char *ret = (char *)outval;
    int vallen = strlen(inval) / 2;

    if (outvallen < vallen)
        return -1;

    for (i = 0; i < vallen; ++i) {
        if (*inval >= '0' && *inval <= '9')
            ret[i] = *inval - '0';
        else
            ret[i] = *inval - 'a' + 10;
        inval++;
        if (*inval >= '0' && *inval <= '9')
            ret[i] |= ((*inval - '0') << 4);
        else
            ret[i] |= ((*inval - 'a' + 10) << 4);
        inval++;
    }

    return 0;
}

#endif // PMI_RT_CODEC_H
