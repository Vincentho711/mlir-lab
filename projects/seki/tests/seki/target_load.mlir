// RUN: seki-opt --seki-target=seki-v1 --seki-attach-target %s | FileCheck %s
// Verify that --seki-attach-target loads the target file and attaches
// #seki.target<...> to the module as an attribute

// CHECK: module attributes
// CHECK-SAME: seki.target = #seki.target<memory = <scratchpadBytes = 8388608
// CHECK-SAME: compute = <macArrayRows = 128, macArrayCols = 128
// CHECK-SAME: isa = <isaVersion = 1, maxTileId = 64

module {}
