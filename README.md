# mlir-lab

A monorepo for learning and experimenting with [MLIR](https://mlir.llvm.org/) (Multi-Level Intermediate Representation). Each subdirectory under `projects/` is an independent experiment covering custom dialects, lowering passes, and transformations.

---

## What is MLIR?

MLIR is a compiler infrastructure framework that is part of the LLVM project. It provides a flexible intermediate representation and a set of reusable abstractions for building compilers, optimisers, and code generators. This repo explores it from first principles — defining custom dialects in TableGen, writing lowering passes, and testing them with FileCheck and GoogleTest.

---

## Prerequisites

The following must be installed on your machine before building.

**Bazelisk**

Bazelisk is a launcher for Bazel that automatically downloads the correct Bazel version (pinned in `.bazelversion`). Install it as `bazel`:

```bash
# Linux
curl -Lo /usr/local/bin/bazel \
  https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
chmod +x /usr/local/bin/bazel
```

**Clang and LLD**

The build is configured to use Clang as the compiler and LLD as the linker. On Ubuntu:

```bash
sudo apt install clang lld
```

Verify both are available:

```bash
clang --version   # expect 15 or newer
ld.lld --version
```

> **Note:** GCC and the GNU linker are not supported. The `.bazelrc` hardcodes `CC=clang`, `CXX=clang++`, and `-fuse-ld=lld`.

---

## Getting started

**1. Clone the repo**

```bash
git clone git@github.com:Vincentho711/mlir-lab.git
cd mlir-lab
```

**2. Build**

```bash
bazel build //projects/...
```

**3. Test**

```bash
bazel test //projects/...
```

That is all. No extra setup scripts, no manual LLVM installation.

---

## First build warning

The first build downloads the LLVM source (~3 GB) and compiles it from scratch. **This takes 30–60 minutes** depending on your machine. Subsequent builds are incremental and take seconds.

Bazel caches everything in `~/.cache/bazel/`. As long as that directory persists, you will never need to recompile LLVM again unless the version changes.

---

## Repository structure

```
mlir-lab/
├── projects/          # Independent MLIR experiments (one subdirectory each)
├── MODULE.bazel       # Bzlmod dependency declaration (LLVM, rules_cc, etc.)
├── .bazelrc           # Build flags: Clang, LLD, C++20
├── .bazelversion      # Pins Bazel to a specific version via bazelisk
└── .github/
    └── workflows/
        └── ci.yml     # CI: build and test all projects on push/PR
```

Each experiment under `projects/` is self-contained with its own `BUILD.bazel`, source files, and tests.

---

## LLVM version

This repo builds against **LLVM 22.1.5** (`llvmorg-22.1.5`), fetched automatically by Bazel. No manual LLVM installation is required.
