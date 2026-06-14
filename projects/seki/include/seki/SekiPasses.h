#ifndef SEKI_PASSES_H
#define SEKI_PASSES_H

#include "mlir/Pass/Pass.h"

// Pulls in generated factory decls like createSekiAttachTargetPass()
// GEN_PASS_REGISTRATION allows each register...() calls mlir::registerPass(factory)
// At which point, Pass::Option<std::string> target{..., "seki-target", ...} is constructed and registers --seki-target
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "seki/SekiPasses.h.inc"

#endif // SEKI_PASSES_H
