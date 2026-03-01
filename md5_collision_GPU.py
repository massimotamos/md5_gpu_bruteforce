#!/usr/bin/env python3
# Property Massimo Tamos
# Use only for educational purposes is allowed.
# By using this script you agree to take all responsibilities for misuse.

import os
import pyopencl as cl
import numpy as np
import time
import string
import hashlib
import sys

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

# ----------------------------
# OpenCL Kernel Code (MD5, single-block, GPU brute force)
# ----------------------------
kernel_code = r"""
#define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))
#define MAX_LEN 24

__constant uint T[64] = {
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
};

__constant uint S[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
};

__kernel void md5_bruteforce_kernel(
    __global const uchar* charset,
    const uint charset_length,
    const uint msg_len,
    const ulong start_index,
    const ulong total,
    __global const uchar* target_hash,
    __global int* found_flag,
    __global uchar* result_plaintext   // length = msg_len
) {
    if (*found_flag) return;

    ulong gid = (ulong)get_global_id(0);
    ulong idx = start_index + gid;
    if (idx >= total) return;

    if (msg_len > MAX_LEN) return;

    // Build plaintext from idx in base charset_length
    uchar plaintext[MAX_LEN] = {0};
<<<<<<< HEAD

    for (uint i = 0; i < prefix_len; i++) {
        plaintext[i] = prefix[i];
    }

=======
>>>>>>> dacf054 (GPU Only)
    ulong temp = idx;

    // Fill from right to left
    for (int pos = (int)msg_len - 1; pos >= 0; pos--) {
        plaintext[(uint)pos] = charset[temp % charset_length];
        temp /= charset_length;
    }

    // MD5 initial state
    uint a0 = 0x67452301;
    uint b0 = 0xefcdab89;
    uint c0 = 0x98badcfe;
    uint d0 = 0x10325476;

    // Single-block message buffer (works up to 55 bytes)
    uchar msg[64] = {0};
    for (uint i = 0; i < msg_len; i++) {
        msg[i] = plaintext[i];
    }
    msg[msg_len] = (uchar)0x80;

    // 64-bit length in bits (little-endian)
    ulong bit_len = ((ulong)msg_len) * 8;
    msg[56] = (uchar)(bit_len & 0xFF);
    msg[57] = (uchar)((bit_len >> 8) & 0xFF);
    msg[58] = (uchar)((bit_len >> 16) & 0xFF);
    msg[59] = (uchar)((bit_len >> 24) & 0xFF);
    msg[60] = (uchar)((bit_len >> 32) & 0xFF);
    msg[61] = (uchar)((bit_len >> 40) & 0xFF);
    msg[62] = (uchar)((bit_len >> 48) & 0xFF);
    msg[63] = (uchar)((bit_len >> 56) & 0xFF);

    // Parse into 16 32-bit words
    uint M[16];
    for (uint i = 0; i < 16; i++) {
        M[i] = ((uint)msg[i * 4]) |
               (((uint)msg[i * 4 + 1]) << 8) |
               (((uint)msg[i * 4 + 2]) << 16) |
               (((uint)msg[i * 4 + 3]) << 24);
    }

    uint A = a0, B = b0, C = c0, D = d0;

    for (uint i = 0; i < 64; i++) {
        uint f, g;
        if (i < 16) {
            f = (B & C) | (~B & D);
            g = i;
        } else if (i < 32) {
            f = (D & B) | (~D & C);
            g = (5 * i + 1) % 16;
        } else if (i < 48) {
            f = B ^ C ^ D;
            g = (3 * i + 5) % 16;
        } else {
            f = C ^ (B | ~D);
            g = (7 * i) % 16;
        }

        f = f + A + T[i] + M[g];
        A = D;
        D = C;
        C = B;
        B = B + LEFTROTATE(f, S[i]);
    }

    A += a0; B += b0; C += c0; D += d0;

    // digest little-endian
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

    // Compare
    int match = 1;
    for (uint i = 0; i < 16; i++) {
        if (hash[i] != target_hash[i]) { match = 0; break; }
    }

    if (match) {
        int res = atomic_cmpxchg(found_flag, 0, 1);
        if (res == 0) {
            for (uint i = 0; i < msg_len; i++) {
                result_plaintext[i] = plaintext[i];
            }
        }
    }
}
"""

def main():
    charset = string.ascii_letters + string.digits + string.punctuation
    charset_len = len(charset)  # 94

    target_string = input(
        "Enter a target string of length 1..10 (chars must be in [A-Za-z0-9 punctuation]): "
    ).rstrip("\n")

    if not (1 <= len(target_string) <= 10):
        print("ERROR: Target length must be between 1 and 10.")
        sys.exit(1)

    bad = [c for c in target_string if c not in charset]
    if bad:
        print(f"ERROR: Target contains characters not in charset: {bad}")
        sys.exit(1)

    msg_len = len(target_string)
    target_hash_hex = hashlib.md5(target_string.encode("ascii")).hexdigest()
    target_hash_bytes = bytes.fromhex(target_hash_hex)

    print(f"Target string:   {target_string}")
    print(f"Target MD5 hash: {target_hash_hex}")
    print(f"Charset length:  {charset_len} (letters+digits+punctuation; no space)")

<<<<<<< HEAD
    # Total length of the brute-forced plaintext
    total_len = len(target_string)

    # Split: CPU prefix, GPU suffix
    # Keep CPU prefix small for usability; for short strings adapt gracefully.
    prefix_len = min(2, total_len)
    suffix_len = total_len - prefix_len
    msg_len = prefix_len + suffix_len  # == total_len

    if msg_len > 24:
        print("ERROR: msg_len exceeds MAX_LEN in kernel.")
        sys.exit(1)

    # Suffix total must fit u64
    suffix_total = pow(charset_len, suffix_len)  # if suffix_len==0 => 1
=======
    # Total candidates for exact-length brute force
    total = pow(charset_len, msg_len)
>>>>>>> dacf054 (GPU Only)
    MAX_U64 = (1 << 64) - 1
    if total > MAX_U64:
        print("ERROR: search space exceeds 64-bit indexing; reduce length or charset.")
        sys.exit(1)

<<<<<<< HEAD
    # OpenCL setup (choose first platform/device like your original)
=======
    # OpenCL setup
>>>>>>> dacf054 (GPU Only)
    platforms = cl.get_platforms()
    if not platforms:
        print("ERROR: No OpenCL platforms found.")
        sys.exit(1)

    device = platforms[0].get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    max_wg = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    print(f"OpenCL device: {device.name}")
    print(f"Max work-group size: {max_wg}")

    # Host arrays
    charset_np = np.frombuffer(charset.encode("ascii"), dtype=np.uint8)
    target_hash_np = np.frombuffer(target_hash_bytes, dtype=np.uint8)
    found_np = np.zeros(1, dtype=np.int32)
    result_np = np.zeros(msg_len, dtype=np.uint8)

    # Buffers
    mf = cl.mem_flags
    charset_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=charset_np)
    target_hash_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_hash_np)
    found_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=found_np)
    result_buf = cl.Buffer(context, mf.WRITE_ONLY, size=msg_len)

<<<<<<< HEAD
    # Build
=======
    # Build + kernel
>>>>>>> dacf054 (GPU Only)
    program = cl.Program(context, kernel_code).build()
    kernel = program.md5_bruteforce_kernel

<<<<<<< HEAD
    # Chunking: keep it reasonable (tune if needed)
    CHUNK = 25_000_000

    start_time = time.time()
    checked_prefixes = 0
    total_prefixes = charset_len ** prefix_len

    # Iterate all prefixes on CPU
    for p_idx in itertools.product(range(charset_len), repeat=prefix_len):
        checked_prefixes += 1
=======
    # Chunking (tune)
    CHUNK = 25_000_000

    start_time = time.perf_counter()
    last_print = start_time
    PRINT_EVERY_SECONDS = 1.0

    start_index = 0
    while start_index < total:
        # early-out check
        cl.enqueue_copy(queue, found_np, found_buf)
        if found_np[0] == 1:
            break

        current = int(min(CHUNK, total - start_index))
        global_size = (current,)
>>>>>>> dacf054 (GPU Only)

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

        # Let the driver choose a valid local size (robust)
        cl.enqueue_nd_range_kernel(queue, kernel, global_size, None)
        queue.finish()

        # check after chunk
        cl.enqueue_copy(queue, found_np, found_buf)
        if found_np[0] == 1:
            cl.enqueue_copy(queue, result_np, result_buf)
            found_plain = result_np.tobytes().decode("ascii", errors="strict")
            elapsed = time.perf_counter() - start_time
            tested = start_index + current
            rate = tested / elapsed if elapsed > 0 else 0.0

<<<<<<< HEAD
            current = int(min(CHUNK, suffix_total - start_index))

            # NDRange: global is number of candidates in this chunk
            global_size = (current,)

            # args:
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
=======
            print("\nFOUND!")
            print(f"Plaintext:  {found_plain}")
            print(f"Time:       {elapsed:.2f}s")
            print(f"Rate:       {_fmt_rate(rate)}")
            return

        # status
        now = time.perf_counter()
        if (now - last_print) >= PRINT_EVERY_SECONDS:
            tested = start_index + current
            elapsed = now - start_time
            rate = tested / elapsed if elapsed > 0 else 0.0
            remaining = total - tested
            eta = remaining / rate if rate > 0 else float("inf")
            pct = (tested / total) * 100.0 if total else 0.0

            sys.stdout.write(
                f"\rProgress: {pct:6.2f}% | Tested: {tested:.3e}/{total:.3e} | "
                f"Rate: {_fmt_rate(rate)} | ETA: {_fmt_secs(eta)}"
>>>>>>> dacf054 (GPU Only)
            )
            sys.stdout.flush()
            last_print = now

<<<<<<< HEAD
            # Reliable across devices: let OpenCL choose local size
            cl.enqueue_nd_range_kernel(queue, kernel, global_size, None)
            queue.finish()

            # check after each chunk
            cl.enqueue_copy(queue, found_np, found_buf)
            if found_np[0] == 1:
                cl.enqueue_copy(queue, result_np, result_buf)
                found_plain = result_np.tobytes().decode("ascii", errors="strict")
                elapsed = time.time() - start_time
                print("\nFOUND!")
                print(f"Plaintext:  {found_plain}")
                print(f"Time:       {elapsed:.2f}s")
                print(f"Prefixes tried: {checked_prefixes}/{total_prefixes}")
                return

            start_index += current

        # progress (coarse)
        if checked_prefixes % 250 == 0:
            pct = (checked_prefixes / total_prefixes) * 100.0
            sys.stdout.write(f"\rPrefix progress: {pct:.2f}%")
            sys.stdout.flush()

    elapsed = time.time() - start_time
=======
        start_index += current

    elapsed = time.perf_counter() - start_time
    rate_final = total / elapsed if elapsed > 0 else 0.0
>>>>>>> dacf054 (GPU Only)
    print("\nNot found.")
    print(f"Time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
