/* -*- C -*-
 *
 * This code is derived from MPICH, which is copyright
 *  (C) 2001 by Argonne National Laboratory.
 *
 * Copyright 2016 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S.  Government
 * retains certain rights in this software.
 *
 * Copyright (c) 2016 Intel Corporation. All rights reserved.
 * This software is available to you under the BSD license.
 *
 * This file is part of the Sandia OpenSHMEM software package. For license
 * information, see the LICENSE file in the top level directory of the
 * distribution.
 *
 */

#ifndef MPL_H
#define MPL_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#define ATTRIBUTE         __attribute__
#define MPI_MAX_PORT_NAME 256

#define MPL_snprintf(...) snprintf(__VA_ARGS__)

/*
 * MPL_strncpy - Copy at most n characters.  Stop once a null is reached.
 *
 * This is different from strncpy, which null pads so that exactly
 * n characters are copied.  The strncpy behavior is correct for many
 * applications because it guarantees that the string has no uninitialized
 * data.
 *
 * If n characters are copied without reaching a null, return an error.
 * Otherwise, return 0.
 *
 * Question: should we provide a way to request the length of the string,
 * since we know it?
 */
/*@ MPL_strncpy - Copy a string with a maximum length

Input Parameters:
+   instr - String to copy
-   maxlen - Maximum total length of 'outstr'

Output Parameters:
.   outstr - String to copy into

    Notes:
    This routine is the routine that you wish 'strncpy' was.  In copying
    'instr' to 'outstr', it stops when either the end of 'outstr' (the
    null character) is seen or the maximum length 'maxlen' is reached.
    Unlike 'strncpy', it does not add enough nulls to 'outstr' after
    copying 'instr' in order to move precisely 'maxlen' characters.
    Thus, this routine may be used anywhere 'strcpy' is used, without any
    performance cost related to large values of 'maxlen'.

    If there is insufficient space in the destination, the destination is
    still null-terminated, to avoid potential failures in routines that neglect
    to check the error code return from this routine.

  Module:
  Utility
  @*/
static inline int MPL_strncpy(char *dest, const char *src, size_t n) {
    char *d_ptr = dest;
    const char *s_ptr = src;
    register int i;

    if (n == 0)
        return 0;

    i = (int)n;
    while (*s_ptr && i-- > 0) {
        *d_ptr++ = *s_ptr++;
    }

    if (i > 0) {
        *d_ptr = 0;
        return 0;
    }
    else {
        /* Force a null at the end of the string (gives better safety
         * in case the user fails to check the error code) */
        dest[n - 1] = 0;
        /* We may want to force an error message here, at least in the
         * debugging version */
        /*printf("failure in copying %s with length %d\n", src, n); */
        return 1;
    }
}

static inline void MPL_exit(int exit_code) {
    exit(exit_code);
}

/*@ MPL_strnapp - Append to a string with a maximum length

Input Parameters:
+   instr - String to copy
-   maxlen - Maximum total length of 'outstr'

Output Parameters:
.   outstr - String to copy into

    Notes:
    This routine is similar to 'strncat' except that the 'maxlen' argument
    is the maximum total length of 'outstr', rather than the maximum
    number of characters to move from 'instr'.  Thus, this routine is
    easier to use when the declared size of 'instr' is known.

  Module:
  Utility
  @*/
static inline int MPL_strnapp(char *dest, const char *src, size_t n) {
    char *d_ptr = dest;
    const char *s_ptr = src;
    register int i;

    /* Get to the end of dest */
    i = (int)n;
    while (i-- > 0 && *d_ptr)
        d_ptr++;
    if (i <= 0)
        return 1;

    /* Append.  d_ptr points at first null and i is remaining space. */
    while (*s_ptr && i-- > 0) {
        *d_ptr++ = *s_ptr++;
    }

    /* We allow i >= (not just >) here because the first while decrements
     * i by one more than there are characters, leaving room for the null */
    if (i >= 0) {
        *d_ptr = 0;
        return 0;
    }
    else {
        /* Force the null at the end */
        *--d_ptr = 0;

        /* We may want to force an error message here, at least in the
         * debugging version */
        return 1;
    }
}

static inline int MPL_internal_error_printf(const char *str, ...) {
    int n;
    va_list list;
    const char *format_str;

    va_start(list, str);
    format_str = str;
    n = vfprintf(stderr, format_str, list);
    va_end(list);

    fflush(stderr);

    return n;
}

#endif // MPL_H
