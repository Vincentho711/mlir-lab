#ifndef SEKI_TILING_PATTERNS_H
#define SEKI_TILING_PATTERNS_H
#include "mlir/IR/PatternMatch.h"

namespace mlir::seki {
void populateCastCommutationPatterns(mlir::RewritePatternSet &patterns);
} // namespace mlir::seki
#endif
