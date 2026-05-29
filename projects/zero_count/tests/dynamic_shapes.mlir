// RUN: zero-count-opt %s | FileCheck %s

// Dynamic tensor: ? means unknown at compile-time.
// CHECK-LABEL: func.func @dynamic_matmul
func.func @dynamic_matmul(%a : tensor<?x?xf32>, %b : tensor<?x?xf32>) -> tensor<?x?xf32> {
    // CHECK: tensor<?x?xf32>, tensor<?x?xf32>
    %c = zero_count.matmul %a, %b : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    func.return %c : tensor<?x?xf32>
}
