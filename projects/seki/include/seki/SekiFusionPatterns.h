#ifndef SEKI_FUSION_PATTERNS_H
#define SEKI_FUSION_PATTERNS_H
#include "mlir/IR/PatternMatch.h"

namespace mlir::seki {
void populateTransposeMatmulFusionPatterns(mlir::RewritePatternSet &patterns);
} // namespace mlir::seki
#endif
