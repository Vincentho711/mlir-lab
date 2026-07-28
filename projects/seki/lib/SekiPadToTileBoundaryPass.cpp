#include "seki/SekiPasses.h"
#include "seki/SekiTargetInfo.h"
#include "seki/SekiOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#define GEN_PASS_DEF_SEKIPADTOTILEBOUNDARYPASS
#include "seki/SekiPasses.h.inc"

namespace {

// Computes how many elements to add to `dim` to reach the next multiple of `tile`.
// E.g. dim=128, tile=128 -> 0
// dim=100, tile=128 -> 28
static int64_t padAmount(int64_t dim, int64_t tile) {
    return (-dim % tile + tile) % tile;
}
}
