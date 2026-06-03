#ifndef SEKI_TARGETS_H
#define SEKI_TARGETS_H

#include "seki/SekiAttrs.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::seki {

// Returns a built-in target attribute by name, e.g. "seki-v1"
SekiTargetAttr getBuiltinTarget(llvm::StringRef name, MLIRContext *ctx);

} // namespace mlir::seki
#endif
