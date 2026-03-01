# HOW-TO: Greenfield Ubuntu Setup for `md5_gpu_bruteforce`

This guide bootstraps a fresh Ubuntu machine to a state where you can clone the repo, switch to the correct branch, install Python deps, and run the MD5 brute force project.

> Assumptions  
> - You have `sudo` rights.  
> - You want to use **SSH** access to GitHub.  
> - Python version is **3.10+** (examples use 3.10 on Ubuntu 22.04).  
> - GPU/OpenCL runtime is already available (drivers installed). This guide installs the **OpenCL headers/dev** packages required for `pyopencl`.

---

## 0) System update + base tooling

```bash
sudo apt update
sudo apt install -y git openssh-client ca-certificates curl
```

(Optional but convenient)
```bash
sudo apt install -y python-is-python3
```

---

## 1) Fix GitHub SSH access (public key auth)

### 1.1 Create an SSH key
```bash
ssh-keygen -t ed25519 -C "<your-user>@<your-GPU-server>" -f ~/.ssh/id_ed25519
```

### 1.2 Start ssh-agent and load the key
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
ssh-add -l
```

### 1.3 Add the public key to GitHub
Print the public key:
```bash
cat ~/.ssh/id_ed25519.pub
```

Copy it and add it to:
**GitHub → Settings → SSH and GPG keys → New SSH key**

### 1.4 Trust GitHub host key (non-interactive)
```bash
ssh-keyscan github.com >> ~/.ssh/known_hosts
```

### 1.5 Verify SSH auth
```bash
ssh -T git@github.com
```

Expected: a success message like “Hi <username>! You've successfully authenticated…”

---

## 2) Clone the repository

From your workspace directory:
```bash
mkdir -p ~/dev
cd ~/dev
git clone git@github.com:massimotamos/md5_gpu_bruteforce.git
cd md5_gpu_bruteforce

---

## 3) Switch from `main` to the feature branch

List branches:
```bash
git branch -a
```

In your case, the remote branch was:
- `remotes/origin/feature_improve_1`

Create a local tracking branch and switch to it:
```bash
git fetch --all --prune
git switch -c feature_improve_1 --track origin/feature_improve_1
```

Verify:
```bash
git branch --show-current
git status
```

(If you truly need a local branch named `feature_one`.)
```bash
git switch -c feature_one
git push -u origin feature_one
```

---

## 4) Python venv support + pip

Install the venv package (required on Ubuntu):
```bash
sudo apt install -y python3-venv python3-pip
```

Create and activate a clean virtual environment:
```bash
cd ~/dev/md5_gpu_bruteforce
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

---

## 5) OpenCL dev headers (needed for `pyopencl` on many systems)

```bash
sudo apt install -y ocl-icd-opencl-dev opencl-headers
```

> Note: This installs headers/dev files. You still need a working OpenCL runtime/driver for your GPU (NVIDIA/AMD/Intel) to actually execute kernels.

---

## 6) Dependencies: `requirements.txt` + lockfile

### 6.1 Generate `requirements.txt` from imports (pipreqs)
Install pipreqs in the venv:
```bash
pip install pipreqs
```

Generate requirements from the codebase imports:
```bash
pipreqs . --force --encoding=utf-8
cat requirements.txt
```

### 6.2 Fix NumPy version for Python 3.10 (if needed)

`pipreqs` may guess a NumPy version that requires Python 3.11+.  
For Python 3.10, pin something compatible (example used in our setup):

```bash
sed -i 's/^numpy==.*/numpy==2.2.6/' requirements.txt
cat requirements.txt
```

### 6.3 Install deps + create a reproducible lock file

Start clean (recommended):
```bash
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

Install:
```bash
pip install -r requirements.txt
```

Create lock file:
```bash
pip freeze > requirements-lock.txt
```

Smoke test imports:
```bash
python -c "import numpy, pyopencl; print('numpy', numpy.__version__); print('pyopencl', pyopencl.__version__)"
```

---

## 7) Git add/commit the dependency files

```bash
git add requirements.txt requirements-lock.txt
git status
git commit -m "Add Python dependencies (requirements + lock)"
git push
```

---

## 8) Run the MD5 brute force (project-specific)

Because the exact entrypoint can vary by repo, use one of the following patterns:

### 8.1 If there is a main Python script
```bash
source venv/bin/activate
python <your_script>.py --help
python <your_script>.py <args>
```

### 8.2 If there is a module entrypoint
```bash
source venv/bin/activate
python -m <package_or_module> --help
python -m <package_or_module> <args>
```

### 8.3 Quick discovery of entrypoints
```bash
ls -la
find . -maxdepth 2 -type f -name "*.py" -print
```

---

## Troubleshooting

### A) `git@github.com: Permission denied (publickey)`
- You **don’t have** a private key in `~/.ssh`, or it’s not added to `ssh-agent`.
- Confirm:
  ```bash
  ls -la ~/.ssh
  ssh-add -l
  ssh -T git@github.com
  ```

### B) `pip install -r requirements.txt` fails on NumPy
- Your `requirements.txt` pins a NumPy that requires a newer Python.
- For Python 3.10, pin something compatible (example):
  ```bash
  sed -i 's/^numpy==.*/numpy==2.2.6/' requirements.txt
  ```

### C) `pyopencl` installs but runtime fails
- Headers are installed, but the **OpenCL runtime/driver** is missing or misconfigured.
- Check OpenCL platforms/devices (requires `clinfo`):
  ```bash
  sudo apt install -y clinfo
  clinfo | head -n 80
  ```

---

## Expected end state
You should have:
- `requirements.txt` (direct deps)
- `requirements-lock.txt` (exact installed deps)
- `venv/` (local virtual environment; typically not committed)
