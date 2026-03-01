# md5_gpu_bruteforce

## Finds a string starting from the md5 hash leveraging on the GPU by using OpenCL

# md5_gpu_bruteforce (GPU / OpenCL)

**Purpose:** Educational security demo that illustrates why **MD5 must not be used for security-relevant hashing** (e.g., password storage, digital signatures, integrity protection against active attackers).

This repository contains a small proof-of-concept that uses **GPU acceleration (OpenCL via PyOpenCL)** to demonstrate how quickly an attacker can search a keyspace and recover short plaintext inputs associated with an MD5 hash (i.e., *practical brute-force feasibility for weak inputs*).

MD5 is also widely considered **cryptographically broken for collision resistance** (i.e., it is feasible to craft two different inputs producing the same MD5 digest), which is one of the reasons MD5 has been deprecated for many security use cases.

---

## Why this exists (audit / security awareness)

The original goal of this project was to provide a **hands-on, management-level demonstration** for a Head of Security:

- **“MD5 is outdated and unsafe”** is often understood only abstractly.
- A short, reproducible demo makes the risk tangible: GPU parallelism makes brute-force search significantly faster than typical CPU-only assumptions.
- The intent is to drive **risk-based decisions**: migrate to modern primitives (e.g., SHA-256/SHA-512 for general hashing; password hashing with a dedicated KDF such as Argon2/bcrypt/scrypt; signatures with modern schemes), and remove MD5 from security architectures.

---

## What this repo *does*

- Demonstrates an **MD5 brute-force search accelerated on the GPU** using **OpenCL** (`md5_collision_GPU.py`).
- Uses a defined character set and a fixed length to search for a plaintext that matches a target MD5 digest (or equivalently, validates a candidate by hashing and comparing).

---

## What this repo *does NOT* do

- It is **not** a “universal MD5 cracker” and it is **not** intended for real-world cracking.
- It is **not** a collision-generation framework.  
  (MD5 collision generation is a separate topic; the high-level security point remains: **MD5 collision resistance is broken**, and MD5 should not be relied upon where collisions matter.)

---

## Concrete guidance: how to choose a “hackable” string

Pick parameters so the expected crack time is **10–60 seconds** on your RTX 5060 Ti, so you can demo it live.

### Expected tries (average-case)

Let:

- Charset size: **C**
- Length: **L**
- Search space: **C^L**

For a random target, the expected number of tries is approximately half the search space:

- **Expected tries ≈ (C^L) / 2**

### Examples (average-case tries)

- **Digits only** (C = 10), **L = 8**  
  Expected tries ≈ 0.5 × 10^8 = **50 million** (often “instant” on GPU)

- **Lowercase letters** (C = 26), **L = 7**  
  Expected tries ≈ 0.5 × 8.0e9 ≈ **4e9** (seconds to minutes depending on implementation)

- **Alphanumeric** (C = 62), **L = 7**  
  Expected tries ≈ 0.5 × 3.5e12 ≈ **1.8e12** (minutes to hours)

- **Full 94-char charset** (letters + digits + punctuation), **L = 7**  
  Expected tries ≈ 0.5 × 6.48e13 ≈ **3.24e13** (hours unless very optimized)

### Practical rule of thumb for a live demo

For a live demo target:
- Use **C = 10..36** and **L = 7..9**, depending on your measured throughput (hashes/sec).

---

## Security & legal disclaimer (restricted use)

**RESTRICTED USE — READ CAREFULLY**

This code is provided **strictly for security training, internal awareness, and defensive verification in controlled environments** (e.g., approved lab exercises, secure coding training, sanctioned demonstrations).

**The author explicitly prohibits any use of this code for:**
- unauthorized access attempts,
- password cracking against systems you do not own or explicitly administer,
- any activity that violates laws, contracts, policies, or ethical guidelines.

By using, copying, or modifying this repository, **you accept full responsibility for compliance** with applicable laws and policies. If you do not agree, **do not use this code**.

---

## Why MD5 is considered broken (high level)

MD5 is no longer appropriate for modern security controls because:

- **Collision resistance is broken**: collisions and even chosen-prefix collisions have been demonstrated in practice (historically enabling serious abuse cases such as forged certificate chains). :contentReference[oaicite:4]{index=4}
- Standards bodies and protocol specifications have moved to **deprecate MD5 in security contexts**, particularly for signatures and modern protocol usage.

---

## Operational safety notes (for defenders)

If you are using this repository for training:
- Use only **synthetic test data** (no real user passwords/hashes).
- Run only on **isolated lab systems**.
- Record an **authorization statement** (scope, owner approval, time window) as evidence of legitimate testing.

---

## Repository contents

- `md5_collision_GPU.py` — Python proof-of-concept using PyOpenCL to run MD5 brute-force search on GPU.

---

## Contact / context

If you are reviewing this as part of an audit / security assessment:
- The project should be treated as a **training artifact**, not as a production component.
- Its value is in demonstrating why **legacy hashing choices create avoidable risk**.
