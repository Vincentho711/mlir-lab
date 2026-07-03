// RUN: seki-opt %s | FileCheck %s --check-prefix=CHECK-ROUNDTRIP

// ---------------------------------------------------------------
// Roundtrip: all six binary functions
// ---------------------------------------------------------------

// CHECK-ROUNDTRIP-LABEL: func.func @ew_binary_add
// CHECK-ROUNDTRIP: seki.ew_binary< add> %{{.*}}, %{{.*}} : tensor<128x128xf32>
func.func @ew_binary_add(%lhs: tensor<128x128xf32>, %rhs: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %out = seki.ew_binary<add> %lhs, %rhs : tensor<128x128xf32>
    return %out : tensor<128x128xf32>
}

// CHECK-ROUNDTRIP-LABEL: func.func @ew_binary_sub
// CHECK-ROUNDTRIP: seki.ew_binary< sub> %{{.*}}, %{{.*}} : tensor<128x128xf32>
func.func @ew_binary_sub(%lhs: tensor<128x128xf32>, %rhs: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %out = seki.ew_binary<sub> %lhs, %rhs : tensor<128x128xf32>
    return %out : tensor<128x128xf32>
}

// CHECK-ROUNDTRIP-LABEL: func.func @ew_binary_mul
// CHECK-ROUNDTRIP: seki.ew_binary< mul> %{{.*}}, %{{.*}} : tensor<128x128xf32>
func.func @ew_binary_mul(%lhs: tensor<128x128xf32>, %rhs: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %out = seki.ew_binary<mul> %lhs, %rhs : tensor<128x128xf32>
    return %out : tensor<128x128xf32>
}

// CHECK-ROUNDTRIP-LABEL: func.func @ew_binary_div
// CHECK-ROUNDTRIP: seki.ew_binary< div> %{{.*}}, %{{.*}} : tensor<128x128xf32>
func.func @ew_binary_div(%lhs: tensor<128x128xf32>, %rhs: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %out = seki.ew_binary<div> %lhs, %rhs : tensor<128x128xf32>
    return %out : tensor<128x128xf32>
}

// CHECK-ROUNDTRIP-LABEL: func.func @ew_binary_max
// CHECK-ROUNDTRIP: seki.ew_binary< max> %{{.*}}, %{{.*}} : tensor<128x128xf32>
func.func @ew_binary_max(%lhs: tensor<128x128xf32>, %rhs: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %out = seki.ew_binary<max> %lhs, %rhs : tensor<128x128xf32>
    return %out : tensor<128x128xf32>
}

// CHECK-ROUNDTRIP-LABEL: func.func @ew_binary_min
// CHECK-ROUNDTRIP: seki.ew_binary< min> %{{.*}}, %{{.*}} : tensor<128x128xf32>
func.func @ew_binary_min(%lhs: tensor<128x128xf32>, %rhs: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %out = seki.ew_binary<min> %lhs, %rhs : tensor<128x128xf32>
    return %out : tensor<128x128xf32>
}

// ---------------------------------------------------------------
// Chained with ew_unary: typical post-matmul epilogue pattern
// ---------------------------------------------------------------

// CHECK-ROUNDTRIP-LABEL: func.func @binary_then_unary
// CHECK-ROUNDTRIP: seki.ew_binary< add>
// CHECK-ROUNDTRIP: seki.ew_unary< relu>
func.func @binary_then_unary(%lhs: tensor<128x128xf32>, %rhs: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %sum  = seki.ew_binary<add> %lhs, %rhs : tensor<128x128xf32>
    %relu = seki.ew_unary<relu> %sum       : tensor<128x128xf32>
    return %relu : tensor<128x128xf32>
}
