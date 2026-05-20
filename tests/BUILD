exports_files(["lit.cfg.py"])

filegroup(
    name = "test_utilities",
    testonly = True,
    srcs = ["lit.cfg.py"],
    data = [
        "//tools:mlir-lab-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:count",
        "@llvm-project//llvm:not",
        "@llvm-project//llvm:split-file",
        "@llvm-project//mlir:mlir-opt",
    ],
    visibility = ["//visibility:public"],
)
