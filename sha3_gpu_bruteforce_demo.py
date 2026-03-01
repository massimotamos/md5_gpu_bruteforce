#!/usr/bin/env python3
# Educational demo: GPU brute-force of SHA3-256 (Keccak-f[1600])
# Use only for educational purposes. You are responsible for any misuse.

import os
import sys
import time
import hashlib
import string

import pyopencl as cl
import numpy as np

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"


def infer_charset_from_target(target: str) -> str:
    # Smallest common attacker charset class that still contains the target
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

    printable = "".join(chr(i) for i in range(32, 127))  # space..~
    if all(c in printable for c in target):
        return printable  # 95

    raise ValueError("Target contains non-printable/non-ASCII characters; cannot infer charset safely.")


def fmt_secs(seconds: float) -> str:
    if seconds == float("inf") or seconds != seconds:
        return "N/A"
    seconds = int(max(0, seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d > 0:
        return f"{d}d {h:02d}h {m:02d}m {s:02d}s"
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:02d}m {s:02d}s"
    return f"{s}s"


def fmt_rate(rate: float) -> str:
    if rate < 1e3:
        return f"{rate:.0f} H/s"
    if rate < 1e6:
        return f"{rate/1e3:.2f} KH/s"
    if rate < 1e9:
        return f"{rate/1e6:.2f} MH/s"
    if rate < 1e12:
        return f"{rate/1e9:.2f} GH/s"
    return f"{rate/1e12:.2f} TH/s"


def pow_u64_limit(base: int) -> int:
    limit = (1 << 64) - 1
    e, v = 0, 1
    while True:
        if v > limit // base:
            return e
        v *= base
        e += 1


KERNEL_TEMPLATE = r"""
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#define MAX_LEN {MAX_LEN}
#define SHA3_256_RATE 136  // bytes (1088 bits)

typedef ulong u64;
typedef uint  u32;
typedef uchar u8;

static inline u64 ROTL64(u64 x, u32 n) {{
    return (x << n) | (x >> (64 - n));
}}

__constant u64 RC[24] = {{
  (u64)0x0000000000000001UL, (u64)0x0000000000008082UL,
  (u64)0x800000000000808aUL, (u64)0x8000000080008000UL,
  (u64)0x000000000000808bUL, (u64)0x0000000080000001UL,
  (u64)0x8000000080008081UL, (u64)0x8000000000008009UL,
  (u64)0x000000000000008aUL, (u64)0x0000000000000088UL,
  (u64)0x0000000080008009UL, (u64)0x000000008000000aUL,
  (u64)0x000000008000808bUL, (u64)0x800000000000008bUL,
  (u64)0x8000000000008089UL, (u64)0x8000000000008003UL,
  (u64)0x8000000000008002UL, (u64)0x8000000000000080UL,
  (u64)0x000000000000800aUL, (u64)0x800000008000000aUL,
  (u64)0x8000000080008081UL, (u64)0x8000000000008080UL,
  (u64)0x0000000080000001UL, (u64)0x8000000080008008UL
}};

__constant u32 RHO[25] = {{
  0,  1, 62, 28, 27,
 36, 44,  6, 55, 20,
  3, 10, 43, 25, 39,
 41, 45, 15, 21,  8,
 18,  2, 61, 56, 14
}};

static inline u64 load64_le_private(const __private u8 *p) {{
    return ((u64)p[0]) |
           ((u64)p[1] << 8) |
           ((u64)p[2] << 16) |
           ((u64)p[3] << 24) |
           ((u64)p[4] << 32) |
           ((u64)p[5] << 40) |
           ((u64)p[6] << 48) |
           ((u64)p[7] << 56);
}}

static inline void store64_le_private(__private u8 *p, u64 v) {{
    p[0] = (u8)(v & 0xFF);
    p[1] = (u8)((v >> 8) & 0xFF);
    p[2] = (u8)((v >> 16) & 0xFF);
    p[3] = (u8)((v >> 24) & 0xFF);
    p[4] = (u8)((v >> 32) & 0xFF);
    p[5] = (u8)((v >> 40) & 0xFF);
    p[6] = (u8)((v >> 48) & 0xFF);
    p[7] = (u8)((v >> 56) & 0xFF);
}}

static inline void keccak_f1600(u64 A[25]) {{
    for (int round = 0; round < 24; round++) {{
        // Theta
        u64 C[5];
        for (int x = 0; x < 5; x++) {{
            C[x] = A[x] ^ A[x+5] ^ A[x+10] ^ A[x+15] ^ A[x+20];
        }}
        u64 D[5];
        for (int x = 0; x < 5; x++) {{
            D[x] = C[(x+4)%5] ^ ROTL64(C[(x+1)%5], 1);
        }}
        for (int y = 0; y < 5; y++) {{
            for (int x = 0; x < 5; x++) {{
                A[x + 5*y] ^= D[x];
            }}
        }}

        // Rho + Pi
        u64 B[25];
        for (int y = 0; y < 5; y++) {{
            for (int x = 0; x < 5; x++) {{
                int idx = x + 5*y;
                u64 v = ROTL64(A[idx], RHO[idx]);
                int nx = y;
                int ny = (2*x + 3*y) % 5;
                B[nx + 5*ny] = v;
            }}
        }}

        // Chi
        for (int y = 0; y < 5; y++) {{
            for (int x = 0; x < 5; x++) {{
                A[x + 5*y] = B[x + 5*y] ^ ((~B[((x+1)%5) + 5*y]) & B[((x+2)%5) + 5*y]);
            }}
        }}

        // Iota
        A[0] ^= RC[round];
    }}
}}

__kernel void sha3_256_bruteforce(
    __global const u8* charset,
    const u32 charset_len,
    const u32 msg_len,
    const ulong start_index,
    const ulong total,
    __global const u8* target_hash,   // 32 bytes
    __global int* found_flag,
    __global u8* result_plaintext     // msg_len bytes
) {{
    if (*found_flag) return;

    ulong gid = (ulong)get_global_id(0);
    ulong idx = start_index + gid;
    if (idx >= total) return;

    if (msg_len > MAX_LEN) return;

    // Build plaintext from idx in base charset_len (right-to-left)
    u8 m[MAX_LEN] = {{0}};
    ulong t = idx;
    for (int pos = (int)msg_len - 1; pos >= 0; pos--) {{
        m[(u32)pos] = charset[t % charset_len];
        t /= charset_len;
    }}

    // Keccak state
    u64 A[25] = {{0}};

    // Build one rate block (136 bytes). msg_len <= 55 so single absorb.
    u8 block[SHA3_256_RATE] = {{0}};
    for (u32 i = 0; i < msg_len; i++) block[i] = m[i];

    // SHA3 domain sep: 0x06, final bit 0x80 at end of rate
    block[msg_len] ^= (u8)0x06;
    block[SHA3_256_RATE - 1] ^= (u8)0x80;

    // Absorb: XOR first 17 lanes (136 bytes)
    for (int lane = 0; lane < 17; lane++) {{
        A[lane] ^= load64_le_private((__private u8*)&block[lane*8]);
    }}

    keccak_f1600(A);

    // Squeeze 32 bytes from first 4 lanes
    u8 out[32];
    for (int i = 0; i < 4; i++) {{
        store64_le_private((__private u8*)&out[i*8], A[i]);
    }}

    int match = 1;
    for (int i = 0; i < 32; i++) {{
        if (out[i] != target_hash[i]) {{ match = 0; break; }}
    }}

    if (match) {{
        int res = atomic_cmpxchg(found_flag, 0, 1);
        if (res == 0) {{
            for (u32 i = 0; i < msg_len; i++) result_plaintext[i] = m[i];
        }}
    }}
}}
"""

KERNEL_24 = KERNEL_TEMPLATE.format(MAX_LEN=24)
KERNEL_55 = KERNEL_TEMPLATE.format(MAX_LEN=55)


def main():
    print("=== SHA3-256 GPU brute-force demo ===")
    target = input("Enter target plaintext (length 1..55): ").rstrip("\n")

    if not (1 <= len(target) <= 55):
        print("ERROR: supported length is 1..55 (single-rate-block demo).")
        sys.exit(1)

    charset = infer_charset_from_target(target)

    target_bytes = target.encode("latin1")
    target_hash = hashlib.sha3_256(target_bytes).digest()
    target_hash_hex = target_hash.hex()

    print(f"Target plaintext: {target}")
    print(f"Target length:    {len(target_bytes)}")
    print(f"Inferred charset: {len(charset)} chars")
    print(f"Target SHA3-256:  {target_hash_hex}")
    print(f"\n")
    print("Note: SHA3-256 is cryptographically stronger than MD5, but it is still a fast hash. "
      "Short/low-entropy secrets can still be brute-forced. For passwords, prefer Argon2/scrypt/bcrypt.")

    charset_bytes = charset.encode("latin1")
    charset_len = len(charset_bytes)
    msg_len = len(target_bytes)

    # u64 indexing limit
    max_len = pow_u64_limit(charset_len)
    if msg_len > max_len:
        print(f"\nERROR: Search space |charset|^len does not fit in 64-bit indexing for GPU-only.")
        print(f"With charset size {charset_len}, max GPU-only length is {max_len}.")
        sys.exit(1)

    total = int(pow(charset_len, msg_len))

    # OpenCL setup
    platforms = cl.get_platforms()
    if not platforms:
        print("ERROR: No OpenCL platforms found.")
        sys.exit(1)

    device = platforms[0].get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    print(f"OpenCL device: {device.name}")
    max_wg = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    print(f"Max work-group size: {max_wg}")

    # Select kernel variant
    kernel_src = KERNEL_24 if msg_len <= 24 else KERNEL_55
    program = cl.Program(context, kernel_src).build()
    kernel = program.sha3_256_bruteforce

    mf = cl.mem_flags
    charset_np = np.frombuffer(charset_bytes, dtype=np.uint8)
    target_hash_np = np.frombuffer(target_hash, dtype=np.uint8)
    found_np = np.zeros(1, dtype=np.int32)
    result_np = np.zeros(msg_len, dtype=np.uint8)

    charset_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=charset_np)
    target_hash_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_hash_np)
    found_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=found_np)
    result_buf = cl.Buffer(context, mf.WRITE_ONLY, size=msg_len)

    # Tuning knobs
    CHUNK = 50_000_000
    CHECK_EVERY = 20
    PRINT_EVERY_SECONDS = 2.0

    start_time = time.perf_counter()
    last_print = start_time

    submitted = 0
    completed = 0
    chunk_count = 0
    last_evt = None

    start_index = 0
    while start_index < total:
        current = int(min(CHUNK, total - start_index))
        global_size = (current,)

        kernel.set_args(
            charset_buf,
            np.uint32(charset_len),
            np.uint32(msg_len),
            np.uint64(start_index),
            np.uint64(total),
            target_hash_buf,
            found_buf,
            result_buf
        )

        # enqueue kernel (no finish)
        last_evt = cl.enqueue_nd_range_kernel(queue, kernel, global_size, None)
        submitted += current
        chunk_count += 1

        # Only print based on COMPLETED (strict)
        now = time.perf_counter()
        if (now - last_print) >= PRINT_EVERY_SECONDS:
            elapsed = now - start_time
            rate = completed / elapsed if elapsed > 0 else 0.0
            remaining = total - completed
            eta = remaining / rate if rate > 0 else float("inf")
            pct = (completed / total) * 100.0 if total else 0.0
            sys.stdout.write(
                f"\rProgress: {pct:6.2f}% | Completed: {completed:.3e}/{total:.3e} | "
                f"Rate: {fmt_rate(rate)} | ETA: {fmt_secs(eta)}"
            )
            sys.stdout.flush()
            last_print = now

        # Occasionally sync/check
        if (chunk_count % CHECK_EVERY) == 0:
            last_evt.wait()
            completed = submitted  # strict completion point

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

        start_index += current

    # final sync/check
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
    main()
