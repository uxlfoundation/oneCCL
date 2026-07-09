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

/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

/* maximum sizes for arrays */
#define PMIU_MAXLINE        1024
#define PMIU_ERROR_MSG_SIZE (PMIU_MAXLINE * 3)
#define PMIU_IDSIZE         32

/* we don't have access to MPIR_Assert and friends here in the PMI code */
#if defined(HAVE_ASSERT_H)
#include <assert.h>
#define PMIU_Assert(expr) assert(expr)
#else
#define PMIU_Assert(expr)
#endif

#if defined HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif // HAVE_ARPA_INET_H

/* prototypes for PMIU routines */
void PMIU_Set_rank(int PMI_rank);

void PMIU_SetServer(void);

void PMIU_printf(int print_flag, const char* fmt, ...);

int PMIU_readline(int fd, char* buf, int max);

int PMIU_writeline(int fd, char* buf);

int PMIU_parse_keyvals(char* st);

void PMIU_dump_keyvals(void);

char* PMIU_getval(const char* keystr, char* valstr, int vallen);

void PMIU_chgval(const char* keystr, char* valstr);
