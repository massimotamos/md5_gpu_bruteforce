#!/usr/bin/env python3
# Property Massimo Tamos
# Use only for educational purposes is allowed.
# By using this script you agree to take all responsibilities for misuse.

import os
import sys
import time
import math
import hashlib
import itertools
import multiprocessing as mp
from typing import Optional

import pyopencl as cl
import numpy as np

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

# ============================
# OpenCL kernel template (two variants differ only by MAX_LEN)
# - Single-block MD5 => msg_len must be <= 55
# - Suffix enumeration indexed by uint64 (ulong)
# ============================
KERNEL_TEMPLATE = r"""
#define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))
#define MAX_LEN {MAX_LEN}

__constant uint T[64] = {{
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,

    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,

    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,

    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
}};

__constant uint S[64] = {{
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
}};

__kernel void md5_suffix_kernel(
    __global const uchar* charset,
    const uint charset_length,
    __global const uchar* prefix,   // may be dummy if prefix_len==0
    const uint prefix_len,
    const uint suffix_len,
    const ulong start_index,
    const ulong suffix_total,
    __global const uchar* target_hash,
    __global int* found_flag,
    __global uchar* result_plaintext // length = prefix_len + suffix_len
) {{
    if (*found_flag) return;

    ulong gid = (ulong)get_global_id(0);
    ulong idx = start_index + gid;
    if (idx >= suffix_total) return;

    uint msg_len = prefix_len + suffix_len;
    if (msg_len > MAX_LEN) return;

    uchar plaintext[MAX_LEN] = {{0}};

    for (uint i = 0; i < prefix_len; i++) {{
        plaintext[i] = prefix[i];
    }}

    ulong temp = idx;
    for (int pos = (int)suffix_len - 1; pos >= 0; pos--) {{
        plaintext[prefix_len + (uint)pos] = charset[temp % charset_length];
        temp /= charset_length;
    }}

    uint a0 = 0x67452301;
    uint b0 = 0xefcdab89;
    uint c0 = 0x98badcfe;
    uint d0 = 0x10325476;

    uchar msg[64] = {{0}};
    for (uint i = 0; i < msg_len; i++) msg[i] = plaintext[i];
    msg[msg_len] = (uchar)0x80;

    ulong bit_len = ((ulong)msg_len) * 8;
    msg[56] = (uchar)(bit_len & 0xFF);
    msg[57] = (uchar)((bit_len >> 8) & 0xFF);
    msg[58] = (uchar)((bit_len >> 16) & 0xFF);
    msg[59] = (uchar)((bit_len >> 24) & 0xFF);
    msg[60] = (uchar)((bit_len >> 32) & 0xFF);
    msg[61] = (uchar)((bit_len >> 40) & 0xFF);
    msg[62] = (uchar)((bit_len >> 48) & 0xFF);
    msg[63] = (uchar)((bit_len >> 56) & 0xFF);

    uint M[16];
    for (uint i = 0; i < 16; i++) {{
        M[i] = ((uint)msg[i * 4]) |
               (((uint)msg[i * 4 + 1]) << 8) |
               (((uint)msg[i * 4 + 2]) << 16) |
               (((uint)msg[i * 4 + 3]) << 24);
    }}

    uint A = a0, B = b0, C = c0, D = d0;

    for (uint i = 0; i < 64; i++) {{
        uint f, g;
        if (i < 16) {{
            f = (B & C) | (~B & D);
            g = i;
        }} else if (i < 32) {{
            f = (D & B) | (~D & C);
            g = (5 * i + 1) % 16;
        }} else if (i < 48) {{
            f = B ^ C ^ D;
            g = (3 * i + 5) % 16;
        }} else {{
            f = C ^ (B | ~D);
            g = (7 * i) % 16;
        }}
        f = f + A + T[i] + M[g];
        A = D; D = C; C = B;
        B = B + LEFTROTATE(f, S[i]);
    }}

    A += a0; B += b0; C += c0; D += d0;

    uchar hash[16];
    hash[0]  = (uchar)(A & 0xFF);
    hash[1]  = (uchar)((A >> 8) & 0xFF);
    hash[2]  = (uchar)((A >> 16) & 0xFF);
    hash[3]  = (uchar)((A >> 24) & 0xFF);
    hash[4]  = (uchar)(B & 0xFF);
    hash[5]  = (uchar)((B >> 8) & 0xFF);
    hash[6]  = (uchar)((B >> 16) & 0xFF);
    hash[7]  = (uchar)((B >> 24) & 0xFF);
    hash[8]  = (uchar)(C & 0xFF);
    hash[9]  = (uchar)((C >> 8) & 0xFF);
    hash[10] = (uchar)((C >> 16) & 0xFF);
    hash[11] = (uchar)((C >> 24) & 0xFF);
    hash[12] = (uchar)(D & 0xFF);
    hash[13] = (uchar)((D >> 8) & 0xFF);
    hash[14] = (uchar)((D >> 16) & 0xFF);
    hash[15] = (uchar)((D >> 24) & 0xFF);

    int match = 1;
    for (uint i = 0; i < 16; i++) {{
        if (hash[i] != target_hash[i]) {{ match = 0; break; }}
    }}

    if (match) {{
        int res = atomic_cmpxchg(found_flag, 0, 1);
        if (res == 0) {{
            for (uint i = 0; i < msg_len; i++) result_plaintext[i] = plaintext[i];
        }}
    }}
}}
"""

KERNEL_24 = KERNEL_TEMPLATE.format(MAX_LEN=24)
KERNEL_55 = KERNEL_TEMPLATE.format(MAX_LEN=55)

def fmt_secs(seconds: float) -> str:
    if seconds == float("inf") or seconds != seconds:
        return "N/A"
    seconds = int(max(0, seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d > 0: return f"{d}d {h:02d}h {m:02d}m {s:02d}s"
    if h > 0: return f"{h:02d}h {m:02d}m {s:02d}s"
    if m > 0: return f"{m:02d}m {s:02d}s"
    return f"{s}s"

def fmt_rate(rate: float) -> str:
    if rate < 1e3: return f"{rate:.0f} H/s"
    if rate < 1e6: return f"{rate/1e3:.2f} KH/s"
    if rate < 1e9: return f"{rate/1e6:.2f} MH/s"
    if rate < 1e12: return f"{rate/1e9:.2f} GH/s"
    return f"{rate/1e12:.2f} TH/s"

def pow_u64_limit(base: int) -> int:
    """max exponent e such that base^e <= 2^64-1"""
    limit = (1 << 64) - 1
    e, v = 0, 1
    while True:
        if v > limit // base:
            return e
        v *= base
        e += 1

def _dedupe_keep_order(s: str) -> str:
    seen = set()
    out = []
    for c in s:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return "".join(out)

# ----------------------------
# CPU multicore prefix generator (only for prefix_len > 0)
# It does NOT run GPU kernels in parallel; it only generates prefixes in parallel.
# ----------------------------
def _int_to_base_bytes(n: int, base: int, length: int, alphabet_bytes: bytes) -> bytes:
    out = bytearray(length)
    for pos in range(length - 1, -1, -1):
        out[pos] = alphabet_bytes[n % base]
        n //= base
    return bytes(out)

def _prefix_worker(start: int, end: int, base: int, length: int, alphabet_bytes: bytes, q: mp.Queue):
    for i in range(start, end):
        q.put(_int_to_base_bytes(i, base, length, alphabet_bytes))
    q.put(None)  # sentinel from this worker

def make_prefix_iterator(prefix_len: int, charset_bytes: bytes, workers: int, queue_max: int = 10000):
    """
    Returns an iterator yielding prefix bytes of length prefix_len.
    If workers <= 1: generates in-process (fast enough for small prefix_len).
    If workers > 1: uses multiprocessing to pre-generate prefixes into a queue.
    """
    base = len(charset_bytes)
    total = base ** prefix_len

    if prefix_len == 0:
        yield b""
        return

    if workers <= 1 or total <= 5000:
        # simple local generation
        for idx in range(total):
            yield _int_to_base_bytes(idx, base, prefix_len, charset_bytes)
        return

    q: mp.Queue = mp.Queue(maxsize=queue_max)
    procs = []
    # partition range
    chunk = (total + workers - 1) // workers
    for w in range(workers):
        s = w * chunk
        e = min(total, (w + 1) * chunk)
        if s >= e:
            break
        p = mp.Process(target=_prefix_worker, args=(s, e, base, prefix_len, charset_bytes, q), daemon=True)
        p.start()
        procs.append(p)

    done = 0
    while done < len(procs):
        item = q.get()
        if item is None:
            done += 1
        else:
            yield item

    for p in procs:
        p.join(timeout=1)

import string

def infer_charset_from_target(target: str) -> str:
    # Most realistic “attacker model”: pick the smallest common charset class
    if all(c in string.digits for c in target):
        return string.digits  # 10
    if all(c in string.ascii_lowercase for c in target):
        return string.ascii_lowercase  # 26
    if all(c in string.ascii_uppercase for c in target):
        return string.ascii_uppercase  # 26
    if all(c in string.ascii_letters for c in target):
        return string.ascii_letters  # 52
    if all(c in (string.ascii_letters + string.digits) for c in target):
        return string.ascii_letters + string.digits  # 62

    # Fallback: printable ASCII 32..126 (space + punctuation)
    printable = "".join(chr(i) for i in range(32, 127))  # 95
    if all(c in printable for c in target):
        return printable

    raise ValueError("Target contains non-printable/non-ASCII characters; cannot infer charset safely.")

def main():
    print("=== MD5 GPU brute-force demo ===")
    target = input("Enter target plaintext (1..35 recommended; <=55 supported): ").rstrip("\n")

    if not (1 <= len(target) <= 55):
        print("ERROR: length must be 1..55 for this single-block MD5 kernel.")
        sys.exit(1)

    # Auto-infer charset from target
    charset = infer_charset_from_target(target)
    charset = "".join(dict.fromkeys(charset))  # de-dup while preserving order (safe) 

    charset = _dedupe_keep_order(charset)
    if len(charset) < 2:
        print("ERROR: charset must contain at least 2 distinct characters.")
        sys.exit(1)

    # Use latin1 so bytes map 1:1 for 0..255; ASCII is a subset
    try:
        target_bytes = target.encode("latin1")
        charset_bytes = charset.encode("latin1")
    except UnicodeEncodeError:
        print("ERROR: target/charset must be latin1 encodable (ASCII recommended).")
        sys.exit(1)

    # Ensure target is reachable with charset
    bad = [c for c in target if c not in charset]
    if bad:
        print(f"ERROR: target contains chars not in charset: {bad}")
        sys.exit(1)

    msg_len = len(target_bytes)
    charset_len = len(charset_bytes)

    target_hash_hex = hashlib.md5(target_bytes).hexdigest()
    target_hash_bytes = bytes.fromhex(target_hash_hex)

    print(f"Target: {target}")
    print(f"Length: {msg_len}")
    print(f"Charset size: {charset_len}")
    print(f"Target MD5: {target_hash_hex}")

    # Decide split so suffix fits in uint64
    max_suffix_len = pow_u64_limit(charset_len)
    suffix_len = min(msg_len, max_suffix_len)  # GPU-only if possible
    prefix_len = msg_len - suffix_len

    # practical cap to avoid insane prefix enumeration
    MAX_PREFIX_LEN = 4
    if prefix_len > MAX_PREFIX_LEN:
        print("\nERROR: Search space does not fit in 64-bit indexing for GPU-only.")
        print(f"To fit GPU suffix in 64-bit, prefix_len would be {prefix_len} (> {MAX_PREFIX_LEN}).")
        print("Reduce length and/or charset size for a live demo.")
        sys.exit(1)

    # sizes
    suffix_total = int(pow(charset_len, suffix_len))  # guaranteed fits in uint64
    prefix_total = int(pow(charset_len, prefix_len)) if prefix_len > 0 else 1
    total_candidates = pow(charset_len, msg_len)      # Python big int OK for display

    mode = "GPU-only" if prefix_len == 0 else f"CPU-prefix({prefix_len}) + GPU-suffix({suffix_len})"
    print(f"Mode: {mode}")
    print(f"Suffix space: {charset_len}^{suffix_len} = {suffix_total:.3e}")
    if prefix_len > 0:
        print(f"Prefix space: {charset_len}^{prefix_len} = {prefix_total:.3e}")
    print(f"Total space:  {charset_len}^{msg_len} = {float(total_candidates):.3e} (avg tries ~ half)")

    # OpenCL device/context
    platforms = cl.get_platforms()
    if not platforms:
        print("ERROR: No OpenCL platforms found.")
        sys.exit(1)

    device = platforms[0].get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    print(f"OpenCL device: {device.name}")

    # Pick kernel variant
    kernel_src = KERNEL_24 if msg_len <= 24 else KERNEL_55
    program = cl.Program(context, kernel_src).build()
    kernel = program.md5_suffix_kernel

    # Buffers
    mf = cl.mem_flags
    charset_np = np.frombuffer(charset_bytes, dtype=np.uint8)
    target_hash_np = np.frombuffer(target_hash_bytes, dtype=np.uint8)
    found_np = np.zeros(1, dtype=np.int32)
    result_np = np.zeros(msg_len, dtype=np.uint8)

    charset_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=charset_np)
    target_hash_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_hash_np)
    found_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=found_np)
    result_buf = cl.Buffer(context, mf.WRITE_ONLY, size=msg_len)

    # prefix buffer: dummy 1 byte if prefix_len==0
    prefix_buf = cl.Buffer(context, mf.READ_ONLY, size=max(1, prefix_len))

    # Tuning knobs (reduce CPU overhead)
    CHUNK = 100_000_000
    CHECK_EVERY = 20
    PRINT_EVERY_SECONDS = 2.0

    # CPU multicore for prefix generation (only if prefix_len>0)
    default_workers = max(1, (os.cpu_count() or 1) // 2)
    if prefix_len > 0:
        try:
            w_in = input(f"CPU workers for prefix generation [1..{os.cpu_count() or 1}] (default {default_workers}): ").strip()
            cpu_workers = int(w_in) if w_in else default_workers
        except ValueError:
            cpu_workers = default_workers
        cpu_workers = max(1, min(cpu_workers, os.cpu_count() or 1))
    else:
        cpu_workers = 1

    # Rate/ETA based on COMPLETED work only
    start_time = time.perf_counter()
    last_print = start_time
    submitted = 0
    completed = 0
    chunk_count = 0
    last_evt = None

    def print_status():
        nonlocal last_print
        now = time.perf_counter()
        if (now - last_print) < PRINT_EVERY_SECONDS:
            return
        elapsed = now - start_time
        rate = completed / elapsed if elapsed > 0 else 0.0
        remaining = float(total_candidates - completed) if total_candidates > completed else 0.0
        eta = remaining / rate if rate > 0 else float("inf")
        pct = (float(completed) / float(total_candidates)) * 100.0 if total_candidates else 0.0
        sys.stdout.write(
            f"\rProgress: {pct:6.2f}% | Completed: {completed:.3e}/{float(total_candidates):.3e} | "
            f"Rate: {fmt_rate(rate)} | ETA: {fmt_secs(eta)}"
        )
        sys.stdout.flush()
        last_print = now

    # Prefix iterator (possibly multicore)
    if prefix_len == 0:
        prefix_iter = [b""]
    else:
        prefix_iter = make_prefix_iterator(prefix_len, charset_bytes, workers=cpu_workers)

    for prefix_bytes in prefix_iter:
        # reset found flag per prefix (simple + correct)
        found_np[0] = 0
        cl.enqueue_copy(queue, found_buf, found_np)

        # write prefix
        if prefix_len > 0:
            cl.enqueue_copy(queue, prefix_buf, np.frombuffer(prefix_bytes, dtype=np.uint8))

        start_index = 0
        while start_index < suffix_total:
            current = int(min(CHUNK, suffix_total - start_index))
            global_size = (current,)

            kernel.set_args(
                charset_buf,
                np.uint32(charset_len),
                prefix_buf,
                np.uint32(prefix_len),
                np.uint32(suffix_len),
                np.uint64(start_index),
                np.uint64(suffix_total),
                target_hash_buf,
                found_buf,
                result_buf
            )

            # enqueue; no finish()
            last_evt = cl.enqueue_nd_range_kernel(queue, kernel, global_size, None)
            submitted += current
            chunk_count += 1

            # Only count as completed when we actually waited
            if (chunk_count % CHECK_EVERY) == 0:
                last_evt.wait()
                completed = submitted

                # check found
                cl.enqueue_copy(queue, found_np, found_buf).wait()
                if found_np[0] == 1:
                    cl.enqueue_copy(queue, result_np, result_buf).wait()
                    found_plain = result_np.tobytes().decode("latin1", errors="strict")
                    elapsed = time.perf_counter() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    print("\nFOUND!")
                    print(f"Plaintext:  {found_plain}")
                    print(f"Time:       {elapsed:.2f}s")
                    print(f"Rate:       {fmt_rate(rate)}")
                    return

                print_status()

            start_index += current

    # final synchronization/check (don’t miss a hit in the last partial chunk)
    if last_evt is not None:
        last_evt.wait()
        completed = submitted
        cl.enqueue_copy(queue, found_np, found_buf).wait()
        if found_np[0] == 1:
            cl.enqueue_copy(queue, result_np, result_buf).wait()
            found_plain = result_np.tobytes().decode("latin1", errors="strict")
            elapsed = time.perf_counter() - start_time
            rate = completed / elapsed if elapsed > 0 else 0.0
            print("\nFOUND!")
            print(f"Plaintext:  {found_plain}")
            print(f"Time:       {elapsed:.2f}s")
            print(f"Rate:       {fmt_rate(rate)}")
            return

    elapsed = time.perf_counter() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0
    print("\nNot found.")
    print(f"Time: {elapsed:.2f}s")
    print(f"Rate: {fmt_rate(rate)}")

if __name__ == "__main__":
    # Multiprocessing on Linux is fine; this guard is still required.
    main()
