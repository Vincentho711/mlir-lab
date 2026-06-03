// RUN: seki-opt --seki-target=%S/data/target_seki_test.mlir --seki-attach-target %s | FileCheck %s
// Verify that --seki-attach-target loads the target file and attaches
// #seki.target<...> to the module as an attribute

// CHECK: module attributes
// CHECK-SAME: seki.target = #seki.target<memory = <scratchpadBytes = 524288

module {}
