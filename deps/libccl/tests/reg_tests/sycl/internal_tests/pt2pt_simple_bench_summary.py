#!/usr/bin/env python3
"""
Summarize pt2pt_simple_bench logs.

Reads pt2pt_simple_bench_latency_mode{0,1}.log (and _bw_mode*) from the
directory given on the command line (or the script directory by default),
diffs mode 0 (LL256 baseline) vs mode 1 (new simple protocol), and prints
a comparison table plus pass/fail against the benchmark plan's criteria.

The ccl_latency / ccl_bw binaries are launched under `mpiexec -l`, which
prefixes every line with `[<rank>] `. Each data row is then:
    [<rank>]  <#bytes>  <repetitions>  <latency or bandwidth>
(latency in us, bandwidth in MB/s). We parse that by regex and key on byte
count, consuming only rank-0 rows since both ranks print the same numbers.
"""

import argparse
import os
import re
import sys

# Plan thresholds. Tweak here if criteria shift.
SMALL_MAX_BYTES = 16 * 1024          # 16 KB
LARGE_MIN_BYTES = 1024 * 1024        # 1 MB
SPEEDUP_SMALL = 1.5                  # simple must be >=1.5x faster <=16 KB
PARITY_LARGE_PCT = 10                # within 10% at 1 MB
BW_RATIO_MIN = 0.90                  # >=90% of LL bandwidth across sweep

# ccl_latency / ccl_bw print rows like:
#   [0]              4              1000             20.70
#   [0]           4096                64            140.92
# columns: bytes, repetitions, value (us for latency, MB/s for bandwidth).
# mpiexec -l prefixes every line with "[<rank>] "; we only consume rank 0
# rows since both ranks print the same numbers.
DATA_ROW = re.compile(
    r"^\[0\]\s+(?P<bytes>\d+)\s+\d+\s+(?P<value>[0-9]+\.[0-9]+)\s*$"
)

def parse_log(path):
    """Return dict: bytes -> measurement (latency in us or BW in MB/s)."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as fp:
        for line in fp:
            m = DATA_ROW.match(line)
            if not m:
                continue
            nbytes = int(m.group("bytes"))
            value = float(m.group("value"))
            out[nbytes] = value
    return out

def _missing_rows(baseline, simple):
    """Return sorted list of byte counts present in one log but not both."""
    return sorted(set(baseline) ^ set(simple))

def compare_latency(baseline, simple):
    """Return list of (bytes, bytes, lat0, lat1, speedup) sorted by bytes.

    Only includes byte counts present in *both* logs — comparison rows
    with one side missing are excluded so the threshold checks below
    don't have to special-case None. Callers should treat missing rows
    (returned separately by _missing_rows) as a hard failure.
    """
    rows = []
    for nbytes in sorted(set(baseline) & set(simple)):
        b0 = baseline[nbytes]
        b1 = simple[nbytes]
        speedup = (b0 / b1) if b1 else None
        rows.append((nbytes, nbytes, b0, b1, speedup))
    return rows

def compare_bw(baseline, simple):
    """Return list of (bytes, bytes, bw0, bw1, ratio) sorted by bytes.

    Same intersection-only semantics as compare_latency.
    """
    rows = []
    for nbytes in sorted(set(baseline) & set(simple)):
        b0 = baseline[nbytes]
        b1 = simple[nbytes]
        ratio = (b1 / b0) if b0 else None
        rows.append((nbytes, nbytes, b0, b1, ratio))
    return rows

def fmt(v, width=10, prec=2):
    return f"{v:>{width}.{prec}f}" if v is not None else " " * (width - 1) + "-"

def human_bytes(n):
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.0f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"

def evaluate_latency(rows):
    """Return (fail_msgs, info_msgs)."""
    fails = []
    infos = []
    # Small regime: every size <= SMALL_MAX_BYTES must meet SPEEDUP_SMALL.
    # Previously we only checked max(speedups), which let one fast point
    # mask regressions at every other small size.
    small = [(nb, s) for (nb, _, b0, b1, s) in rows
             if nb <= SMALL_MAX_BYTES and s is not None]
    if not small:
        fails.append("no small-regime latency data to evaluate")
    else:
        worst_nb, worst_s = min(small, key=lambda r: r[1])
        infos.append(
            f"small-regime worst speedup (<= {SMALL_MAX_BYTES}B): "
            f"{worst_s:.2f}x @ {human_bytes(worst_nb)}"
        )
        for nb, s in small:
            if s < SPEEDUP_SMALL:
                fails.append(
                    f"small regime @ {human_bytes(nb)}: speedup {s:.2f}x "
                    f"< required {SPEEDUP_SMALL}x"
                )

    # Large regime: at least one row must be >= LARGE_MIN_BYTES, and every
    # such row must be within PARITY_LARGE_PCT of LL.
    large_seen = False
    for nbytes, _, b0, b1, _ in rows:
        if nbytes < LARGE_MIN_BYTES or b0 is None or b1 is None:
            continue
        large_seen = True
        pct = 100.0 * (b1 - b0) / b0
        infos.append(
            f"large regime @ {human_bytes(nbytes)}: "
            f"simple {b1:.2f}us vs LL {b0:.2f}us ({pct:+.1f}%)"
        )
        if pct > PARITY_LARGE_PCT:
            fails.append(
                f"large regime @ {human_bytes(nbytes)}: simple {pct:+.1f}% "
                f"> allowed +{PARITY_LARGE_PCT}%"
            )
    if not large_seen:
        fails.append(
            f"no latency rows >= {human_bytes(LARGE_MIN_BYTES)}; cannot "
            f"evaluate large-regime parity criterion"
        )
    return fails, infos

def evaluate_bw(rows):
    fails = []
    infos = []
    ratios = [r for (_, _, b0, b1, r) in rows if r is not None]
    if not ratios:
        infos.append("no bandwidth data")
        return fails, infos
    worst = min(ratios)
    infos.append(f"bandwidth worst-case ratio simple/LL: {worst:.2f}")
    if worst < BW_RATIO_MIN:
        fails.append(
            f"bandwidth: worst ratio {worst:.2f} < required {BW_RATIO_MIN}"
        )
    return fails, infos

def print_latency(rows):
    print(f"{'size':>10}  {'bytes':>10}  {'LL (us)':>10}  {'simple (us)':>12}  {'speedup':>8}")
    for nbytes, _, b0, b1, s in rows:
        human = human_bytes(nbytes)
        s_str = f"{s:.2f}x" if s is not None else "    -"
        print(f"{human:>10}  {nbytes:>10d}  {fmt(b0)}  {fmt(b1, width=12)}  {s_str:>8}")

def print_bw(rows):
    print(f"{'size':>10}  {'bytes':>10}  {'LL (MB/s)':>10}  {'simple':>10}  {'ratio':>8}")
    for nbytes, _, b0, b1, r in rows:
        human = human_bytes(nbytes)
        r_str = f"{r:.2f}" if r is not None else "   -"
        print(f"{human:>10}  {nbytes:>10d}  {fmt(b0)}  {fmt(b1)}  {r_str:>8}")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", nargs="?",
                    default=os.path.dirname(os.path.abspath(__file__)),
                    help="directory containing pt2pt_simple_bench_*.log")
    args = ap.parse_args()

    d = args.directory
    any_fail = False

    lat0 = parse_log(os.path.join(d, "pt2pt_simple_bench_latency_mode0.log"))
    lat1 = parse_log(os.path.join(d, "pt2pt_simple_bench_latency_mode1.log"))
    if lat0 and lat1:
        print("=== Latency (mode0 = LL256 baseline, mode1 = simple) ===")
        rows = compare_latency(lat0, lat1)
        print_latency(rows)
        fails, infos = evaluate_latency(rows)
        # Asymmetric size sets between modes are a real failure: the rows
        # in only one log are silently dropped from comparison, which would
        # let a truncated run pass the threshold gate.
        missing = _missing_rows(lat0, lat1)
        if missing:
            fails.append(
                "latency size sets differ between modes; sizes present in "
                f"only one log: {missing}"
            )
        print()
        for m in infos: print(f"  info: {m}")
        for m in fails: print(f"  FAIL: {m}")
        any_fail = any_fail or bool(fails)
    else:
        # Empty/missing logs almost always mean the bench failed to run (binary
        # didn't load, args rejected, etc). Treat as a failure rather than a
        # silent pass — a perf gate that passes on no data is worse than useless.
        print("FAIL: latency logs missing or unparseable")
        any_fail = True

    bw0 = parse_log(os.path.join(d, "pt2pt_simple_bench_bw_mode0.log"))
    bw1 = parse_log(os.path.join(d, "pt2pt_simple_bench_bw_mode1.log"))
    if bw0 and bw1:
        print()
        print("=== Bandwidth (mode0 = LL256 baseline, mode1 = simple) ===")
        rows = compare_bw(bw0, bw1)
        print_bw(rows)
        fails, infos = evaluate_bw(rows)
        missing = _missing_rows(bw0, bw1)
        if missing:
            fails.append(
                "bandwidth size sets differ between modes; sizes present in "
                f"only one log: {missing}"
            )
        print()
        for m in infos: print(f"  info: {m}")
        for m in fails: print(f"  FAIL: {m}")
        any_fail = any_fail or bool(fails)
    else:
        print("FAIL: bandwidth logs missing or unparseable")
        any_fail = True

    # Correctness logs: grep for "ERROR" / "mismatch" as a fail signal.
    for mode in (0, 1):
        path = os.path.join(d, f"pt2pt_simple_bench_correctness_mode{mode}.log")
        if not os.path.isfile(path):
            continue
        with open(path) as fp:
            txt = fp.read()
        bad = re.findall(r"(?i)(error|mismatch|fail)", txt)
        print(f"correctness mode{mode}: {'OK' if not bad else f'{len(bad)} suspicious lines'}")
        if bad:
            any_fail = True

    print()
    print("OVERALL:", "FAIL" if any_fail else "PASS")
    sys.exit(1 if any_fail else 0)

if __name__ == "__main__":
    main()
