// RUN: bazel run //project/zero_count:zero-count-opt -- %s | FileCheck %s

func.func @test_count_zeros(%x : i32) -> i32 {
    // CHECK: zero_count.count_zeros
    %result = zero_count.count_zeros %x : i32
    func.return %result : i32
}
