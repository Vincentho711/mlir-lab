"""Macros for defining lit tests."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@rules_python//python:defs.bzl", "py_test")

def _ensure_lit_cfg():
    """Copy tests/lit.cfg.py into this package's runfiles directory.

    lit discovers test suites by walking up from the test file looking for
    lit.cfg.py. Generating a copy in each calling package keeps
    tests/lit.cfg.py as the single source of truth while making discovery
    work for .mlir files anywhere in the project tree.
    """
    if not native.existing_rule("lit_cfg"):
        copy_file(
            name = "lit_cfg",
            src = "//tests:lit.cfg.py",
            out = "lit.cfg.py",
            testonly = True,
        )

def lit_test(name = None, src = None, tools = [], data = [], size = "small", tags = None):
    """Run a single .mlir file through lit.

    The .mlir file must contain RUN: and CHECK: directives. lit executes the
    RUN: line(s) and pipes output through FileCheck.

    Args:
      name:  test target name (defaults to <src>.test)
      src:   source .mlir file (required)
      tools: extra Bazel targets (binaries) placed in PATH for RUN: lines.
             Use this for project-specific *-opt binaries. Each target lands
             under _main/projects/<project>/ in the runfiles tree, which
             tests/lit.cfg.py adds to PATH automatically.
      size:  Bazel test size — use "medium" for files that invoke mlir-cpu-runner
      tags:  forwarded to py_test
    """
    if not src:
        fail("src must be specified")
    name = name or src + ".test"

    _ensure_lit_cfg()

    native.filegroup(
        name = name + ".filegroup",
        srcs = [src],
    )

    py_test(
        name = name,
        size = size,
        args = ["-v", paths.join(native.package_name(), src)],
        data = ["//tests:test_utilities", ":lit_cfg", name + ".filegroup"] + tools + data,
        srcs = ["//bazel:lit_wrapper.py"],
        main = "//bazel:lit_wrapper.py",
        deps = ["@llvm-project//llvm:lit"],
        python_version = "PY3",
        tags = tags,
    )

def glob_lit_tests(tools = [], data = [], size = "small"):
    """Generate a lit_test for every .mlir file in the calling package.

    Args:
      tools: extra Bazel targets (binaries) placed in PATH for RUN: lines.
             Forwarded to every lit_test in this package.
      data:  extra Bazel targets (data files) placed in runfiles for RUN: lines.
             Forwarded to every lit_test in this package.
      size:  Bazel test size applied to all discovered tests
    """
    for src in native.glob(["*.mlir"]):
        lit_test(src = src, tools = tools, data = data, size = size)
