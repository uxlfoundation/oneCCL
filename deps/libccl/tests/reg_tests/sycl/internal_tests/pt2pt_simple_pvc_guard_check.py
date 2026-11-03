#!/usr/bin/env python3
"""Compare two latency logs and fail if any size bucket differs by > 5%.

Used by pt2pt_simple_pvc_guard.sh to confirm the simple protocol is a
no-op on non-ARC hardware.

Usage:
    pt2pt_simple_pvc_guard_check.py <log_dir>
"""

import os
import re
import sys

THRESHOLD_PCT = 5.0

# mpiexec -l prefixes timing rows with "[<rank>] "; rows look like:
#   [0]              4              1000             20.70
DATA = re.compile(
    r"^\[0\]\s+(?P<bytes>\d+)\s+\d+\s+(?P<v>[0-9]+\.[0-9]+)\s*$"
)


def load(path):
    results = {}
    with open(path) as fp:
        for line in fp:
            m = DATA.match(line)
            if m:
                results[int(m.group("bytes"))] = float(m.group("v"))
    return results


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <log_dir>", file=sys.stderr)
        sys.exit(2)

    log_dir = sys.argv[1]
    a = load(os.path.join(log_dir, "pt2pt_simple_pvc_guard_mode0.log"))
    b = load(os.path.join(log_dir, "pt2pt_simple_pvc_guard_mode1.log"))

    if not a or not b:
        print(f"FAIL: parsed {len(a)} mode0 / {len(b)} mode1 rows; check the regex / logs")
        sys.exit(1)

    common = sorted(set(a) & set(b))
    if not common:
        print("FAIL: no byte counts present in both logs")
        sys.exit(1)

    fail = False
    print(f"{'bytes':>10} {'mode0(us)':>10} {'mode1(us)':>10} {'delta%':>8}")
    for c in common:
        v0, v1 = a[c], b[c]
        d_pct = 100.0 * (v1 - v0) / v0
        flag = "  OK"
        if abs(d_pct) > THRESHOLD_PCT:
            flag = " FAIL"
            fail = True
        print(f"{c:>10d} {v0:>10.2f} {v1:>10.2f} {d_pct:>+7.1f}%{flag}")

    print("OVERALL:", "FAIL" if fail else "PASS")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
