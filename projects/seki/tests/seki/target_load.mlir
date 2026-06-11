// RUN: seki-opt --seki-target=seki-v1 --seki-attach-target %s | FileCheck %s
// Verify that --seki-attach-target loads the target file and attaches
// #seki.target<...> to the module as an attribute

// CHECK: module attributes
// CHECK-SAME: seki.target = #seki.target<memory = <scratchpadBytes = 262144

module {}
