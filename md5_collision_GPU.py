# Property Massimo Tamos
# Use only for educational purposes is allowed.
# By using this script you agree to take all responsilibites for misuse.

import os
import pyopencl as cl
import numpy as np
import time
import string
import hashlib
import math
import sys

# Enable OpenCL compiler output for debugging
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

# OpenCL Kernel Code (complete MD5 implementation)
kernel_code = """
#define F(x, y, z) ((x & y) | (~x & z))
#define G(x, y, z) ((x & z) | (y & ~z))
#define H(x, y, z) (x ^ y ^ z)
#define I(x, y, z) (y ^ (x | ~z))
#define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))

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

__kernel void md5_bruteforce_kernel(__global const uchar* charset,
                                    const uint charset_length,
                                    const uint max_length,
                                    __global const uchar* target_hash,
                                    __global int* result_found,
                                    __global uchar* result_plaintext,
                                    const ulong start_index,
                                    const ulong total_combinations) {
    ulong gid = start_index + get_global_id(0);

    if (*result_found)
        return;

    if (gid >= total_combinations)
        return;

    // Generate plaintext based on gid
    uchar plaintext[8] = {0}; // Max length of 8 characters
    ulong temp = gid;

    for (int pos = max_length - 1; pos >= 0; pos--) {
        plaintext[pos] = charset[temp % charset_length];
        temp /= charset_length;
    }

    // Prepare the initial MD5 state
    uint a0 = 0x67452301;
    uint b0 = 0xefcdab89;
    uint c0 = 0x98badcfe;
    uint d0 = 0x10325476;

    // Prepare padded message
    uchar msg[64] = {0};
    for (uint i = 0; i < max_length; i++) {
        msg[i] = plaintext[i];
    }
    msg[max_length] = 0x80;

    // Append original length in bits
    uint bit_len = max_length * 8;
    msg[56] = bit_len & 0xFF;
    msg[57] = (bit_len >> 8) & 0xFF;
    msg[58] = (bit_len >> 16) & 0xFF;
    msg[59] = (bit_len >> 24) & 0xFF;

    // Process message in 512-bit chunks
    uint M[16];
    for (uint i = 0; i < 16; i++) {
        M[i] = ((uint)msg[i * 4]) |
               (((uint)msg[i * 4 + 1]) << 8) |
               (((uint)msg[i * 4 + 2]) << 16) |
               (((uint)msg[i * 4 + 3]) << 24);
    }

    // Main MD5 algorithm loop
    uint A = a0, B = b0, C = c0, D = d0;

    for (uint i = 0; i < 64; i++) {
        uint F, g;
        if (i < 16) {
            F = (B & C) | (~B & D);
            g = i;
        } else if (i < 32) {
            F = (D & B) | (~D & C);
            g = (5 * i + 1) % 16;
        } else if (i < 48) {
            F = B ^ C ^ D;
            g = (3 * i + 5) % 16;
        } else {
            F = C ^ (B | ~D);
            g = (7 * i) % 16;
        }
        F = F + A + T[i] + M[g];
        A = D;
        D = C;
        C = B;
        B = B + LEFTROTATE(F, S[i]);
    }

    // Add state to the previous values
    A += a0;
    B += b0;
    C += c0;
    D += d0;

    // Compute hash
    uchar hash[16];
    hash[0] = A & 0xFF;
    hash[1] = (A >> 8) & 0xFF;
    hash[2] = (A >> 16) & 0xFF;
    hash[3] = (A >> 24) & 0xFF;
    hash[4] = B & 0xFF;
    hash[5] = (B >> 8) & 0xFF;
    hash[6] = (B >> 16) & 0xFF;
    hash[7] = (B >> 24) & 0xFF;
    hash[8] = C & 0xFF;
    hash[9] = (C >> 8) & 0xFF;
    hash[10] = (C >> 16) & 0xFF;
    hash[11] = (C >> 24) & 0xFF;
    hash[12] = D & 0xFF;
    hash[13] = (D >> 8) & 0xFF;
    hash[14] = (D >> 16) & 0xFF;
    hash[15] = (D >> 24) & 0xFF;

    // Compare computed hash with target_hash
    int match = 1;
    for (uint i = 0; i < 16; i++) {
        if (hash[i] != target_hash[i]) {
            match = 0;
            break;
        }
    }

    // If match found, write plaintext to result_plaintext and set result_found
    if (match) {
        int res = atomic_cmpxchg(result_found, 0, 1);
        if (res == 0) {
            for (uint i = 0; i < max_length; i++) {
                result_plaintext[i] = plaintext[i];
            }
        }
    }
}
"""

def brute_force_md5_opencl(target_string, charset, max_length):
    # Compute MD5 hash of the target string
    target_hash_hex = hashlib.md5(target_string.encode('utf-8')).hexdigest()
    target_hash_bytes = bytes.fromhex(target_hash_hex)
    print(f"Target string: {target_string}")
    print(f"Target MD5 hash: {target_hash_hex}")

    # Create OpenCL context and queue
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices()
    device = devices[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Get the maximum work group size for the device
    max_work_group_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    print(f"Max work group size: {max_work_group_size}")

    # Prepare data
    charset_np = np.array([ord(c) for c in charset], dtype=np.uint8)
    target_hash_np = np.frombuffer(target_hash_bytes, dtype=np.uint8)
    result_found_np = np.array([0], dtype=np.int32)
    result_plaintext_np = np.zeros(max_length, dtype=np.uint8)

    # Create buffers
    mf = cl.mem_flags
    charset_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=charset_np)
    target_hash_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_hash_np)
    result_found_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=result_found_np)
    result_plaintext_buf = cl.Buffer(context, mf.WRITE_ONLY, size=64)

    # Compile the kernel
    program = cl.Program(context, kernel_code).build()
    kernel = program.md5_bruteforce_kernel

    # Calculate total number of combinations
    total_combinations = len(charset) ** max_length

    # Adjust chunk size to fit the GPU memory
    chunk_size = min(1_000_000_000, total_combinations)
    num_chunks = math.ceil(total_combinations / chunk_size)

    start_time = time.time()
    found = False
    last_printed_progress = 0

    for chunk in range(num_chunks):
        start_index = chunk * chunk_size
        end_index = min(start_index + chunk_size, total_combinations)
        current_chunk_size = end_index - start_index

        # Reset result_found buffer
        result_found_np[0] = 0
        cl.enqueue_copy(queue, result_found_buf, result_found_np)

        # Set kernel arguments
        kernel.set_arg(0, charset_buf)
        kernel.set_arg(1, np.uint32(len(charset)))
        kernel.set_arg(2, np.uint32(max_length))
        kernel.set_arg(3, target_hash_buf)
        kernel.set_arg(4, result_found_buf)
        kernel.set_arg(5, result_plaintext_buf)
        kernel.set_arg(6, np.uint64(start_index))
        kernel.set_arg(7, np.uint64(total_combinations))

        # Set global and local work sizes
        global_work_size = (current_chunk_size,)

        # If local_work_size is too large, let OpenCL determine the best size
        local_work_size = (256,) if 256 <= max_work_group_size else None

        try:
            # Execute the kernel
            cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size)
            queue.finish()
        except cl.LogicError as e:
            print(f"Error: {e}")
            print(f"Reducing local work size. Trying auto mode.")
            local_work_size = None
            cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size)
            queue.finish()

        # Check if the result was found
        cl.enqueue_copy(queue, result_found_np, result_found_buf)
        if result_found_np[0]:
            cl.enqueue_copy(queue, result_plaintext_np, result_plaintext_buf)
            found_plaintext = ''.join(map(chr, result_plaintext_np)).strip('\x00')
            end_time = time.time()
            print(f"\nFound plaintext: {found_plaintext}")
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            found = True
            break

        # Calculate and print progress
        progress = ((chunk + 1) * chunk_size / total_combinations) * 100
        if progress - last_printed_progress >= 1:  # Print every 1% increment
            sys.stdout.write(f"\rProgress: {progress:.2f}%")
            sys.stdout.flush()
            last_printed_progress = progress

    if not found:
        end_time = time.time()
        print("\nPlaintext not found.")
        print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    target_string = input("Enter the target string: ").strip()
    charset = string.ascii_lowercase
    max_length = len(target_string)
    brute_force_md5_opencl(target_string, charset, max_length)
