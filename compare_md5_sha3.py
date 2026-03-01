#!/usr/bin/env python3
import subprocess
import re
import csv
import sys
from dataclasses import dataclass
from typing import Optional, List

MD5_SCRIPT = "./md5_collision_GPU.py"
SHA3_SCRIPT = "./sha3_gpu_bruteforce_demo.py"

# Parse lines like:
# Time:       14.01s
# Rate:       1.07 GH/s
TIME_RE = re.compile(r"^Time:\s+([0-9.]+)s$", re.IGNORECASE)
RATE_RE = re.compile(r"^Rate:\s+([0-9.]+)\s+([KMGTP]?H/s)$", re.IGNORECASE)
LEN_RE = re.compile(r"^Target length:\s+(\d+)\s*$", re.IGNORECASE)
CHARSET_RE = re.compile(r"^Inferred charset:\s+(\d+)\s+chars\s*$", re.IGNORECASE)
MD5_RE = re.compile(r"^Target MD5:\s+([0-9a-f]{32})\s*$", re.IGNORECASE)
SHA3_RE = re.compile(r"^Target SHA3-256:\s+([0-9a-f]{64})\s*$", re.IGNORECASE)


@dataclass
class Result:
    algo: str
    length: int
    charset_size: Optional[int]
    hash_hex: Optional[str]
    time_s: Optional[float]
    rate_hps: Optional[float]
    rate_str: Optional[str]


def rate_to_hps(val: float, unit: str) -> float:
    unit = unit.upper()
    mult = {
        "H/S": 1.0,
        "KH/S": 1e3,
        "MH/S": 1e6,
        "GH/S": 1e9,
        "TH/S": 1e12,
        "PH/S": 1e15,
    }.get(unit)
    if mult is None:
        raise ValueError(f"Unknown unit: {unit}")
    return val * mult


def hps_to_str(hps: Optional[float]) -> str:
    if hps is None:
        return "N/A"
    if hps >= 1e12:
        return f"{hps/1e12:.2f} TH/s"
    if hps >= 1e9:
        return f"{hps/1e9:.2f} GH/s"
    if hps >= 1e6:
        return f"{hps/1e6:.2f} MH/s"
    if hps >= 1e3:
        return f"{hps/1e3:.2f} KH/s"
    return f"{hps:.0f} H/s"


def run_script(script_path: str, algo: str, plaintext: str) -> Result:
    p = subprocess.run(
        [script_path],
        input=(plaintext + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    lines = p.stdout.decode("utf-8", errors="replace").splitlines()

    length = len(plaintext)
    charset_size = None
    hash_hex = None
    time_s = None
    rate_hps = None
    rate_str = None

    for raw in lines:
        line = raw.strip()

        m = LEN_RE.match(line)
        if m:
            length = int(m.group(1))

        m = CHARSET_RE.match(line)
        if m:
            charset_size = int(m.group(1))

        m = MD5_RE.match(line)
        if m:
            hash_hex = m.group(1).lower()

        m = SHA3_RE.match(line)
        if m:
            hash_hex = m.group(1).lower()

        m = TIME_RE.match(line)
        if m:
            time_s = float(m.group(1))

        m = RATE_RE.match(line)
        if m:
            val = float(m.group(1))
            unit = m.group(2)
            rate_hps = rate_to_hps(val, unit)
            rate_str = f"{val:g} {unit}"

    return Result(
        algo=algo,
        length=length,
        charset_size=charset_size,
        hash_hex=hash_hex,
        time_s=time_s,
        rate_hps=rate_hps,
        rate_str=rate_str,
    )


def print_markdown_table(rows: List[Result]) -> None:
    # group by length
    by_len = {}
    for r in rows:
        by_len.setdefault(r.length, {})[r.algo] = r

    lengths = sorted(by_len.keys())

    print("| Len | Charset | MD5 Rate | MD5 Time(s) | SHA3-256 Rate | SHA3-256 Time(s) | Speedup (MD5/SHA3) |")
    print("|---:|---:|---:|---:|---:|---:|---:|")

    for L in lengths:
        md5 = by_len[L].get("MD5")
        sha3 = by_len[L].get("SHA3-256")

        charset = md5.charset_size if md5 and md5.charset_size is not None else (sha3.charset_size if sha3 else None)
        charset_s = str(charset) if charset is not None else "N/A"

        md5_rate = hps_to_str(md5.rate_hps if md5 else None)
        sha3_rate = hps_to_str(sha3.rate_hps if sha3 else None)

        md5_time = f"{md5.time_s:.2f}" if (md5 and md5.time_s is not None) else "N/A"
        sha3_time = f"{sha3.time_s:.2f}" if (sha3 and sha3.time_s is not None) else "N/A"

        speedup = "N/A"
        if md5 and sha3 and md5.rate_hps and sha3.rate_hps and sha3.rate_hps > 0:
            speedup = f"{(md5.rate_hps / sha3.rate_hps):.2f}x"

        print(f"| {L} | {charset_s} | {md5_rate} | {md5_time} | {sha3_rate} | {sha3_time} | {speedup} |")


def main():
    base = input("Enter a base string (we will benchmark prefixes length 1..N): ").rstrip("\n")
    if not base:
        print("ERROR: base string must not be empty.")
        sys.exit(1)

    try:
        max_len_in = input(f"Max length N (1..{len(base)}) [default {len(base)}]: ").strip()
        max_len = int(max_len_in) if max_len_in else len(base)
    except ValueError:
        print("ERROR: N must be an integer.")
        sys.exit(1)

    max_len = max(1, min(max_len, len(base)))

    out_csv = "bench_md5_vs_sha3.csv"
    all_rows: List[Result] = []

    for L in range(1, max_len + 1):
        s = base[:L]
        print(f"\n=== Length {L} / {max_len} | plaintext='{s}' ===")

        print("Running MD5...")
        r1 = run_script(MD5_SCRIPT, "MD5", s)
        print(f"  MD5:  rate={hps_to_str(r1.rate_hps)} time={r1.time_s if r1.time_s is not None else 'N/A'}s charset={r1.charset_size}")

        print("Running SHA3-256...")
        r2 = run_script(SHA3_SCRIPT, "SHA3-256", s)
        print(f"  SHA3: rate={hps_to_str(r2.rate_hps)} time={r2.time_s if r2.time_s is not None else 'N/A'}s charset={r2.charset_size}")

        all_rows.extend([r1, r2])

    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algo", "length", "charset_size", "hash_hex", "time_s", "rate_hps"])
        for r in all_rows:
            w.writerow([
                r.algo,
                r.length,
                r.charset_size if r.charset_size is not None else "",
                r.hash_hex if r.hash_hex is not None else "",
                f"{r.time_s:.6f}" if r.time_s is not None else "",
                f"{r.rate_hps:.6f}" if r.rate_hps is not None else "",
            ])

    print("\n=== Markdown table (copy/paste into README.md) ===\n")
    print_markdown_table(all_rows)

    print(f"\nCSV saved to: {out_csv}")


if __name__ == "__main__":
    main()
